$sizes = @(200, 400, 800, 1200, 1600, 2000)
$exe = ".\Release\matrix_mult.exe"

"Size,Time_us,Time_ms,MFLOPS" | Out-File results.csv

foreach ($n in $sizes) {
    Write-Host "Running N=$n ..."
    & $exe -r $n results_$n.txt

    if (Test-Path results_$n.txt) {
        $txt = Get-Content results_$n.txt -Raw
        $us = [regex]::Match($txt, '(\d+) мкс').Groups[1].Value
        $ms = [regex]::Match($txt, '(\d+\.?\d*) мс').Groups[1].Value
        $mflops = [regex]::Match($txt, '(\d+\.?\d*) MFLOPS').Groups[1].Value
        "$n,$us,$ms,$mflops" | Out-File results.csv -Append
    } else {
        "$n,FAIL,FAIL,FAIL" | Out-File results.csv -Append
    }
}
Write-Host "Done. Results in results.csv"