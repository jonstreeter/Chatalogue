"""Windows process ownership for the optional native Phonon worker.

Suspend before job assignment: a venv launcher can otherwise spawn its real
Python child before assignment, leaving that child outside the owned job.
https://learn.microsoft.com/windows/win32/procthread/process-creation-flags
"""
import ctypes
from ctypes import wintypes as w
import os
import subprocess


class _BasicLimits(ctypes.Structure):
    _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                ("flags", w.DWORD), ("min_working_set", ctypes.c_size_t),
                ("max_working_set", ctypes.c_size_t), ("active_processes", w.DWORD),
                ("affinity", ctypes.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in
                ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [("basic", _BasicLimits), ("io", _IoCounters),
                ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]


class _ThreadEntry(ctypes.Structure):
    _fields_ = [("size", w.DWORD), ("usage", w.DWORD), ("id", w.DWORD),
                ("owner", w.DWORD), ("base_priority", w.LONG), ("delta_priority", w.LONG), ("flags", w.DWORD)]


class OwnedPhononProcess:
    def __init__(self, command, *, env, stderr):
        if os.name != "nt":
            raise RuntimeError("Integrated Phonon process ownership currently requires Windows")
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([ctypes.c_void_p, w.LPCWSTR], w.HANDLE),
            "SetInformationJobObject": ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD], w.BOOL),
            "AssignProcessToJobObject": ([w.HANDLE, w.HANDLE], w.BOOL),
            "IsProcessInJob": ([w.HANDLE, w.HANDLE, ctypes.POINTER(w.BOOL)], w.BOOL),
            "CloseHandle": ([w.HANDLE], w.BOOL),
            "CreateToolhelp32Snapshot": ([w.DWORD, w.DWORD], w.HANDLE),
            "Thread32First": ([w.HANDLE, ctypes.POINTER(_ThreadEntry)], w.BOOL),
            "Thread32Next": ([w.HANDLE, ctypes.POINTER(_ThreadEntry)], w.BOOL),
            "OpenThread": ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            "ResumeThread": ([w.HANDLE], w.DWORD),
            "OpenProcess": ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            "WaitForSingleObject": ([w.HANDLE, w.DWORD], w.DWORD),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.kernel, name)
            function.argtypes, function.restype = arguments, result
        self.job = self.kernel.CreateJobObjectW(None, None)
        self.process = None
        self.tracked = []
        if not self.job:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            limits = _ExtendedLimits()
            limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not self.kernel.SetInformationJobObject(self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise ctypes.WinError(ctypes.get_last_error())
            self.process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                text=True, encoding="utf-8", bufsize=1, env=env,
                creationflags=subprocess.CREATE_NO_WINDOW | 0x4,  # CREATE_SUSPENDED
            )
            if not self.kernel.AssignProcessToJobObject(self.job, int(self.process._handle)):
                self.process.terminate()  # Still suspended: it has no descendants or model.
                raise ctypes.WinError(ctypes.get_last_error())
            self._resume_primary_thread()
        except BaseException:
            self.close()
            raise

    def _resume_primary_thread(self):
        snapshot = self.kernel.CreateToolhelp32Snapshot(0x4, 0)  # TH32CS_SNAPTHREAD
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            entry = _ThreadEntry()
            entry.size = ctypes.sizeof(entry)
            found = self.kernel.Thread32First(snapshot, ctypes.byref(entry))
            while found:
                if entry.owner == self.process.pid:
                    thread = self.kernel.OpenThread(0x2, False, entry.id)  # THREAD_SUSPEND_RESUME
                    if not thread:
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        if self.kernel.ResumeThread(thread) == 0xFFFFFFFF:
                            raise ctypes.WinError(ctypes.get_last_error())
                        return
                    finally:
                        self.kernel.CloseHandle(thread)
                found = self.kernel.Thread32Next(snapshot, ctypes.byref(entry))
            raise RuntimeError("Owned Phonon process primary thread was not found")
        finally:
            self.kernel.CloseHandle(snapshot)

    def track_worker(self, pid: int):
        # Open and verify the actual interpreter, not just the venv launcher.
        handle = self.kernel.OpenProcess(0x100000 | 0x1000, False, pid)  # SYNCHRONIZE | QUERY_LIMITED_INFORMATION
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        member = w.BOOL()
        if not self.kernel.IsProcessInJob(handle, self.job, ctypes.byref(member)) or not member.value:
            self.kernel.CloseHandle(handle)
            raise RuntimeError("Phonon interpreter is outside its owned process job")
        self.tracked.append(handle)

    def close(self):
        if self.job:
            if not self.kernel.CloseHandle(self.job):
                raise ctypes.WinError(ctypes.get_last_error())
            self.job = None
        for handle in self.tracked:
            # INFINITE is the Windows API sentinel, not an invented deadline.
            if self.kernel.WaitForSingleObject(handle, 0xFFFFFFFF) != 0:
                raise RuntimeError("Could not confirm owned Phonon interpreter exit")
            self.kernel.CloseHandle(handle)
        self.tracked.clear()
        if self.process:
            self.process.wait()
            for stream in (self.process.stdin, self.process.stdout):
                if stream:
                    stream.close()
