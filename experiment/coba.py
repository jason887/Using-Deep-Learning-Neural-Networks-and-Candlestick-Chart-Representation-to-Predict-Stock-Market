blocks = [2, 2, 2, 2]
for stage_id, iterations in enumerate(blocks):
    print("iterations : {}".format(iterations))
    for block_id in range(iterations):
        print("stage_id : {}".format(stage_id))
