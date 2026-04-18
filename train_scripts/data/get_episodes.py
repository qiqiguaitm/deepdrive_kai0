import json, sys
split_id = int(sys.argv[1])
path = f"/vePFS/tim/workspace/deepdive_kai0/kai0/data/Task_A/split_episodes_{split_id}.json"
episodes = json.load(open(path))
print(" ".join(map(str, episodes)))
