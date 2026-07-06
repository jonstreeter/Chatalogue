param(
    [Parameter(Mandatory = $true)]
    [string]$Url,

    [Parameter(Mandatory = $true)]
    [string]$BackendLog,

    [Parameter(Mandatory = $true)]
    [string]$PostgresLog,

    [Parameter(Mandatory = $true)]
    [int64]$PostgresLogStartOffset,

    [int]$TimeoutSeconds = 900
)

$ErrorActionPreference = 'SilentlyContinue'
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$lastRecoveryNotice = ''

while ((Get-Date) -lt $deadline) {
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300) {
            exit 0
        }
    } catch {
        # Backend is not ready yet.
    }

    if (Test-Path $BackendLog) {
        try {
            $tail = (Get-Content -Path $BackendLog -Tail 120 -ErrorAction SilentlyContinue) -join [Environment]::NewLine
            if (
                ($tail -match 'Traceback \(most recent call last\)') -or
                ($tail -match 'EmbeddedPostgresError') -or
                ($tail -match 'ConnectionTimeout') -or
                ($tail -match 'connection timeout expired') -or
                ($tail -match 'did not become query-ready in time') -or
                ($tail -match 'pre-existing shared memory block is still in use') -or
                ($tail -match 'error while attempting to bind on address') -or
                ($tail -match '^No Python at ')
            ) {
                exit 2
            }
        } catch {
            # Ignore transient log-read failures while backend is writing.
        }
    }

    if (Test-Path $PostgresLog) {
        try {
            $content = Get-Content -Path $PostgresLog -Raw -Encoding UTF8 -ErrorAction SilentlyContinue
            if ($null -ne $content -and $content.Length -gt $PostgresLogStartOffset) {
                $start = [Math]::Min($PostgresLogStartOffset, $content.Length)
                $recent = $content.Substring($start)
                $recoverLines = $recent -split "`r?`n" | Where-Object {
                    $_ -match 'syncing data directory \(fsync\), elapsed time:|database system is ready to accept connections|automatic recovery in progress'
                }
                $message = $recoverLines | Select-Object -Last 1
                if ($message) {
                    $message = $message.Trim()
                    if ($message -ne $lastRecoveryNotice) {
                        Write-Host ('  [postgres] ' + $message)
                        $lastRecoveryNotice = $message
                    }
                }
            }
        } catch {
            # Ignore transient log-read failures while postgres is writing.
        }
    }

    Start-Sleep -Milliseconds 800
}

exit 1
