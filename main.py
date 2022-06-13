import Input
import Output
import Brain

inputs = Input.Input()
outputs = Output.Output()
brain = Brain.brain()

# Initialize ==========================================================================================================
initialize_param = 100
batch = []
for _ in range(initialize_param):
    observation = inputs.observe()
    action = brain.predict(observation)
    outputs.act(action)

    batch.append([observation, action])  # add reward
# =====================================================================================================================
while True:
    if not brain.done_training:
        observation = inputs.observe()
        action = brain.predict(observation)
        outputs.act(action)

        batch.append([observation, action]) # add reward

    else:
        brain.train(batch)
        batch = []
