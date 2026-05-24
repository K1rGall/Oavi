param(
    [string]$MicDevice = "Microphone",
    [int]$SampleRate = 16000,
    [int]$SecondsPerWord = 2,
    [int]$PhoneSeconds = 12
)

$ErrorActionPreference = "Stop"

$root = Join-Path (Get-Location) "audio\raw"
$alphabetDir = Join-Path $root "alphabet"
New-Item -ItemType Directory -Force -Path $alphabetDir | Out-Null

$items = @("0","1","2","3","4","5","6","7","8","9","+")

Write-Host "Проверьте имя микрофона командой:"
Write-Host "ffmpeg -list_devices true -f dshow -i dummy"
Write-Host ""
Write-Host "Используемый источник: $MicDevice"
Write-Host ""

foreach ($symbol in $items) {
    $outFile = Join-Path $alphabetDir "$symbol.wav"
    Write-Host "Произнесите: $symbol"
    Start-Sleep -Seconds 1
    ffmpeg -y -f dshow -i "audio=$MicDevice" -ac 1 -ar $SampleRate -t $SecondsPerWord $outFile | Out-Null
    Write-Host "Сохранено: $outFile"
    Start-Sleep -Milliseconds 500
}

$phoneFile = Join-Path $root "phone.wav"
Write-Host ""
Write-Host "Теперь произнесите номер телефона подряд (без сотен и десятков)."
Start-Sleep -Seconds 1
ffmpeg -y -f dshow -i "audio=$MicDevice" -ac 1 -ar $SampleRate -t $PhoneSeconds $phoneFile | Out-Null
Write-Host "Сохранено: $phoneFile"

$expected = Read-Host "Введите эталонную последовательность для phone_expected.txt (например 9031574)"
if (-not [string]::IsNullOrWhiteSpace($expected)) {
    Set-Content -Path (Join-Path $root "phone_expected.txt") -Value $expected -Encoding UTF8
    Write-Host "Сохранено: audio/raw/phone_expected.txt"
}

Write-Host ""
Write-Host "Записи готовы. Запустите:"
Write-Host "py -3.11 scripts/run_pipeline.py"

