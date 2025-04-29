# Save this as run_scrapper.ps1

# Loop from 1 to 17
for ($i = 1; $i -le 26; $i++) {
    # Build the filename
    $htmlFile = ".\$i.html"
    
    # Print which file is being processed
    Write-Output "Processing $htmlFile..."
    
    # Run the command
    python .\scrapper.py --csv .\evaluation_results.csv --images .\images\ $htmlFile

    # Sleep for 30 seconds
    Start-Sleep -Seconds 30
}