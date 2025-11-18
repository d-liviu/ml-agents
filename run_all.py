import subprocess, time



# 1- open unity
# 2- pick the Environment you want, select the scene you want 
# 3- when you select, go to File -> Build Settings -> Build
# 4- select the folder where you want to save the Environment executable
# 5- after building, copy the path of the executable and paste it below
ENV_PATH = "C:\\Users\\dadoi\\Desktop\\MLProject\\ml-agents\\Project\\Builds\\UnityEnvironment.exe"


# List of (run_id, config_file_path) 
CONFIGS = [
    ("WormTestTwo", "C:\\Users\\dadoi\\Desktop\\MLProject\\ml-agents\\config\\ppo\\Worm\\WormTestTwo.yaml"),
    ("WormTest", "C:\\Users\\dadoi\\Desktop\\MLProject\\ml-agents\\config\\ppo\\Worm\\WormTest.yaml"),
    ("Worm", "C:\\Users\\dadoi\\Desktop\\MLProject\\ml-agents\\config\\ppo\\Worm\\Worm.yaml"),
]

for run_id, cfg in CONFIGS:
    cmd = [
        "mlagents-learn",
        cfg,
        f"--run-id={run_id}",
        f"--env={ENV_PATH}",
        "--no-graphics",
        "--train",
        "--force"
    ]
    print(f"Starting run: {run_id}")
    subprocess.run(cmd, check=True)
    print(f" Finished: {run_id}")
    time.sleep(5)

print(" All runs done!")