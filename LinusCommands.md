# Commands for Server 
1. Connect (local)
`ssh laiv3@grace.cas.mcmaster.ca`

2. Upload folder `scp -r LOCALPATH laiv3@grace...:~/ (local)`
A whole folder (note -r for recursive). Quote paths that contain spaces.
`scp -r "C:\Users\laiv3\Code\Summer 2026 Research\SAMPLEFOLDER" laiv3@grace.cas.mcmaster.ca:~/`

3. A single file
`scp myfile.py laiv3@grace.cas.mcmaster.ca:~/SAMPLEFILE.md/`
`scp "C:\Users\vanes\Downloads\Fourth Year\Research GitHUb\Methods Comparison\Hyperparameter Tuning\Embedding Matrix\bottleneck_annotator_sweep.py" "laiv3@grace.cas.mcmaster.ca:/u50/laiv3/Methods Comparison/Embedding Matrix/"`

4. Download results `scp "laiv3@grace...:~/SAMPLEFOLDER/*.csv" . (local)`
`scp "laiv3@grace.cas.mcmaster.ca:~/SAMPLEFOLDER/SAMPLEFILE.md" .`

5. See what's in home	ls -la ~ (server)
6. Activate env (server, every shell)
    ` conda activate vllm `
    Check GPUs (server)	
    `nvidia-smi `

7. `python bottleneck_annotator_sweep.py --data_dir "../dataset"`
