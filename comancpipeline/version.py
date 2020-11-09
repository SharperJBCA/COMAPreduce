import subprocess
git_command = "git log -n 1 --pretty='%H - %cd'"
__version__  = subprocess.run(git_command,shell=True,capture_output=True).stdout.decode().split(' ')[0]
