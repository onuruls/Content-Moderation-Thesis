import argparse
import sys
import os

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from cmt.eval.runner import eval_run
from cmt.utils.io_utils import load_yaml

def main():
    parser = argparse.ArgumentParser(description="Content Moderation Thesis - Evaluation Runner")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to evaluation config YAML")
    parser.add_argument("--run_name", type=str, default=None, help="Override run_name in config")
    args = parser.parse_args()
    
    cfg = load_yaml(args.config)
    
    if args.run_name:
        cfg["run_name"] = args.run_name
        
    print(f"Starting evaluation: {cfg.get('run_name')}")
    eval_run(cfg)

if __name__ == "__main__":
    main()
