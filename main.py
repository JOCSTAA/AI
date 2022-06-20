import Input
import Output
import Brain

lstm_cnt = 5
inputs = Input.Input()
outputs = Output.Output()
brain = Brain.Brain(lstm_timestep=lstm_cnt)

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
    if brain.training:
        observation = inputs.observe()
        action = brain.predict(observation) # try to keep around 40ms or less
        outputs.act(action)

        batch.append([observation, action]) # add reward

    else:
        brain.update_model()
        brain.training = False

        #DO ALL THIS IN UPDAT MODEL

