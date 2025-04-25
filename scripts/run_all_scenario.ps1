param(
    [string]$scenario_dir = "D:/project/scenario_runner/manual_records/*.log"
)
Get-ChildItem -Path $scenario_dir | ForEach-Object {
    Write-Output "Running scenario: $_" 
    python .\carla_replay.py -e -n -f $_.FullName
}

Get-ChildItem -Path $scenario_dir | ForEach-Object {
    Write-Output "Running scenario: $_" 
    python .\carla_replay.py -e -f $_.FullName
}