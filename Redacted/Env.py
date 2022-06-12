from Redacted.Inputs import stream_inputs
class make():
    def __init__(self, stream_time, image_shape = (224,224)):
        self.stream_time = stream_time
        self.image_shape = image_shape

        self.audio_input, self. = stream_inputs(stream_time)

        try:
            g = goal_state[0][0]
            self.goal_state = np.array(goal_state)

        except:
            g = 0

        self.goal_image = make_image(self.goal_state, self.CAD_path, self.iso_details, image_size= image_shape)
        self.allowable_error = allowable_error
        self.reward_style = reward_style
        self.action_style = action_style
        self.observation_style = observation_style
        if self.action_style.lower() == "discrete":
            self.action_size = 12
        elif self.action_style.lower() == "contineous":
            self.action_size = 6
        else:
            print("wrong action entry in env reset")



    def interact(self, action, increment = 1):
        self.current_image = make_image(self.state, self.CAD_path, self.iso_details)
        if self.action_style.lower() == "discrete":
            new_state = act(action, self.state, increment = increment)

        elif self.action_style.lower() == "contineous":
            new_state = c_act(action, self.state)

        else:print("wrong action style entry in env.interact")

        new_image = make_image(new_state, self.CAD_path, self.iso_details)

        if self.reward_style.lower() == "crude":
            rew, done, fail = crude_reward(self.goal_state, self.state, new_state, self.current_image, new_image, allowable_error=self.allowable_error)

        elif self.reward_style.lower() == "image":
            rew, done, fail = image_reward(self.goal_image, self.current_image, new_image, allowable_error=self.allowable_error)

        elif self.reward_style.lower() == "sparse":
            rew, done, fail = sparse_reward(self.goal_state, self.state, new_state, self.current_image, new_image, allowable_error=self.allowable_error)
        else: print("wrong reward style in env.interact")

        self.state = new_state

        if self.observation_style.lower() == "dif_image": new_state = dif_prep(self.goal_image, self.current_image)
        elif self.observation_style.lower() == "stack_image":new_state = stack_prep(self.goal_image, self.current_image)
        elif self.observation_style.lower() == "state_array": new_state = new_state
        else: print("wrong observation style in env", self.observation_style)

        return new_state, rew, done, fail

    def reset(self, variance = 20):
        self.state = rand_reset(variance, self.goal_state)
        if self.observation_style.lower() == "dif_image":
            self.current_image = make_image(self.state, self.CAD_path, self.iso_details)
            new_state = dif_prep(self.goal_image, self.current_image)
            return new_state

        elif self.observation_style.lower() == "stack_image":
            self.current_image = make_image(self.state, self.CAD_path, self.iso_details)
            new_state = stack_prep(self.goal_image, self.current_image)
            return new_state

        elif self.observation_style.lower() == "state_array":
            return self.state
        else:
            print("wrong observation style")


    def show_state_array(self):
        return self.state


    def action_space(self):
        if self.action_style.lower() == "discrete":
            #print("move xyz direction up or down or rot move xyz direction cw or acw")
            return self.action_size

        else:
            #print("move and rot xyz direction - val")
            return self.action_size

    def action_sample(self):
        if self.action_style.lower() == "discrete":
            return np.random.randint(0, self.action_size)

        else:
            return np.random.randint(0, self.action_size)

    def show_progress(self, diff = False):
        if diff:
            imshow(np.vstack((self.current_image, self.goal_image, np.subtract(self.goal_image, self.current_image))))
        else:
            imshow(np.vstack((self.current_image, self.goal_image)))

    def change_state_to(self, state):
        self.state = state

    def show_current_image(self):
        show(self.current_image)

