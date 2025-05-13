# Save this as run_scrapper.ps1

# Loop from 1 to 17
for ($i = 32; $i -le 44; $i++) {
    # Build the filename
    $htmlFile = ".\$i.html"
    
    # Print which file is being processed
    Write-Output "Processing $htmlFile..."
    
    # Run the command
    try {
        python .\scrapper.py --csv .\evaluation_results.csv --images .\images\ $htmlFile
        if ($LASTEXITCODE -ne 0) {
            throw "Python script failed with exit code $LASTEXITCODE"
        }
    } catch {
        Write-Error "An error occurred: $_"
        exit 1
    }

    # Sleep for 30 seconds
    Start-Sleep -Seconds 30
}