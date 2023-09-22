import numpy as np
import tensorflow as tf
import random
import os

_MODEL_FILE_LOCATION="models/saved_models/model_checkpoint"


def convert_wonder_output_to_image(wonder_output, shapes):
    """
    Takes in output of neural network and converts them to eyes and ears images

    :param wonder_output: nn_ouput [batch, sequence, output_vectors]
    :return: eye_images[batch, sequence, l, w, h], ear_images[batch, sequence, l, w, h]
    """
    eyes_input_shape, ears_input_shape = shapes
    batch_size, sequence_length, _ = wonder_output.shape
    eyes_vector_size = np.prod(eyes_input_shape)

    eye_images = wonder_output[:, :, :eyes_vector_size].reshape((batch_size, sequence_length) + eyes_input_shape)
    ear_images = wonder_output[:, :, eyes_vector_size:].reshape((batch_size, sequence_length) + ears_input_shape)

    return eye_images, ear_images

def convert_images_to_wonder_output(eyes, ears):
    """
    Takes in eyes and ears images and converts them to a flattened one dimension layer of neurons

    :param eyes: [batch, sequence, l , w, h]
    :param ears: [batch, sequence, l , w, h]
    :return: [batch, sequence, neurons]
    """
    batch, sequence_length, _, _, _ = eyes.shape

    eyes_flattened = eyes.reshape((batch, sequence_length, -1))
    ears_flattened = ears.reshape((batch, sequence_length, -1))
    output_vector = np.concatenate((eyes_flattened, ears_flattened), axis=-1)

    return output_vector

def train_ppo_loss(epoch, 
                   model, 
                   state, 
                   old_predictions,
                   loss_clip_parameter=0.2
                   ):
    old_samples, old_log_probs, advantages = old_predictions
    for _ in range(epoch):
        with tf.GradientTape() as tape:
            dist_new = model(state)
            new_log_probs = dist_new.log_prob(old_samples)

            ratio_per_sequence = tf.math.exp(new_log_probs - old_log_probs)
            ratio = tf.reduce_mean(ratio_per_sequence)

            surrogate1 = ratio * advantages
            surrogate2 = tf.clip_by_value(ratio, 1 - loss_clip_parameter,
                                            1 + loss_clip_parameter) * advantages

            loss = -tf.math.minimum(surrogate1, surrogate2)
            loss = tf.math.reduce_mean(loss)

        params = model.trainable_variables
        grads = tape.gradient(loss, params)
        model.optimizer.apply_gradients(zip(grads, params))

def train_mse_loss(epoch, 
                   model, 
                   state, 
                   old_predictions,
                   ):
    for _ in range(epoch):
        with tf.GradientTape() as tape:
            predictions = model(state)

            loss = tf.square(predictions - old_predictions)
            loss = 0.5 * tf.reduce_mean(loss)

        params = model.trainable_variables
        grads = tape.gradient(loss, params)
        model.optimizer.apply_gradients(zip(grads, params))

def train_model(model, sleep=False):
    if sleep:
        memory_read = model.memory.read_LT
    else:
        memory_read = model.memory.read_ST

    for unit in memory_read():
        eyes, ears, action_samples, old_log_probs, advantages, old_values, new_eyes, new_ears = unit

        observations = (eyes, ears)
        actor_state = critic_state = observations

        critic_old_predictions = old_values
        actor_old_predictions = (action_samples, old_log_probs, advantages)

        actor, critic, wonder = model.model_in_training.actor, model.model_in_training.critic, model.model_in_training.wonder

        train_mse_loss(model.epoch, critic, critic_state, critic_old_predictions)
        train_ppo_loss(model.epoch, actor, actor_state, actor_old_predictions)

        if model.add_wonder:
            wonder_state = [observations, action_samples]
            wonder_old_predictions = convert_images_to_wonder_output(new_eyes, new_ears)
            train_mse_loss(model.epoch, wonder, wonder_state, wonder_old_predictions)


    # USE THE CURRENT MODEL TO PREDICT THE REAL TIME STATES
    # TRY TO KEEP THIS TIME OPTIMIZED, THIS IS DONE IN REAL TIME

def predict(model, observation):
    """
    USE THE CURRENT MODEL TO PREDICT THE REAL TIME STATES.

    TRY TO KEEP THIS TIME OPTIMIZED, THIS IS DONE IN REAL TIME

    param observation:
    :return: action
    """
    eyes = observation[0]
    ears = observation[1]
    state = (eyes[np.newaxis, :], ears[np.newaxis, :])  # batches up observation, turns it into a one unit batch

    value = model.current_model.critic(state)
    if model.add_wonder:
        _, dist = think(model, state)
    else:
        dist = model.current_model.actor(state)

    action_sample = dist.sample()
    log_probs = dist.log_prob(action_sample)

    # IMPLEMENT REWARD ALGORITHM!
    if model.reward_function:
        reward = model.reward_function(state)
    else: reward = 1

    unit = [eyes, ears, action_sample[0].numpy(), log_probs[0].numpy(), reward, value]
    model.memory.write(unit)

    action = action_sample.numpy()[0]
    return action

def think(model, observation):
    """
    USE THE CURRENT MODEL TO PREDICT THE REAL TIME STATES TAKING ACCOUNT PREDICTIONS OF FUTURE STATES.

    SIMILAR TO PREDICT BUT USES THE WONDER MODEL, WHICH LEARNS TRANSITIONS, SUMS UP ALL THE PREDICTED ACTIONS AND
    TAKES THE ACTION WITH THE HIGHEST PREDICTED VALUE

    :param observation:
    :return: action
    """

    def get_confidence(model_output) -> float:  # IMPLEMENT CONFIDENCE
        """
        Derives the confidence of a model i.e. how sure the model is that the input will lead to that output
        higher = better chance

        param model_output:
        :return:
        """

        return 0
    
    def get_action_likelihood(action_prob_distribution, taken_action) -> float:
        """
        Calculates a measure of how likely the sample is according to the distribution.
        A higher log-likelihood indicates a higher likelihood of the sample being generated by the distribution.
        Higher likelihood = lower chance of exploring other options.

        :param action_prob_distribution: Probability distribution of actions
        :param taken_action: The action that was taken
        :return: prob_to_explore_wider: value from 0 - 1
        """
        log_prob = action_prob_distribution.log_prob(taken_action)
        average_probability = tf.reduce_mean(tf.exp(log_prob)).numpy()

        return average_probability
        return 1 - average_probability # actual return but it currently will while explore_wider() run forever#TODO

    def get_bool_from_probability(random_variable: float) -> bool:
        """
        Returns true or false given a random variable

        param random_variable:
        :return: bool
        """

        return random.random() < random_variable

    def explore_future_actions(curr_state) -> tuple:
        """
        Recursively explores different action paths and outputs the path with the highest predicted reward.

        :param curr_state: The current state
        :return: The best action path's value and the corresponding action distribution
        """
        explore_wider = True
        values = []
        shapes = model.eyes_input_shape, model.ears_input_shape

        while explore_wider:
            dist = model.current_model.actor(curr_state)
            action = dist.sample()
            new_observation = model.current_model.wonder([curr_state, action])
            new_observation = convert_wonder_output_to_image(new_observation, shapes)

            curr_value = model.current_model.critic(new_observation)

            certainty = get_confidence(new_observation)
            exploration = get_action_likelihood(dist, action)

            explore_wider = get_bool_from_probability(exploration)
            explore_deeper = get_bool_from_probability(certainty)

            if explore_deeper:
                values.append((curr_value + explore_future_actions(new_observation)[0], dist))

            else:
                values.append((curr_value, dist))

        values.sort(key=lambda x: -x[0])
        return values[0]

    return explore_future_actions(observation)

def update_model(model):
    """
    PREVIOUS BATCH IS DONE TRAINING, TRAIN THE NEW BATCH:
    UPDATE THE MODEL FIRST THOUGH
    SO THAT THE CURRENT MODEL BEING USED TO PREDICT IS THE LATEST COPY
    """
    if model.memory.length() >= model.min_batch_required_to_train:
        update_model_weights(model)
        train_model(model)

def update_model_weights(model):
    """APPLIES LEARNT MODEL ATTRIBUTES TO CURRENT MODEL"""

    model.current_model.actor.set_weights(model.model_in_training.actor.get_weights())
    model.current_model.wonder.set_weights(model.model_in_training.wonder.get_weights())
    model.current_model.critic.set_weights(model.model_in_training.critic.get_weights())

def instantiate_model_state(model):
    """
    Creates a sample of the state that would be used to train the model
    This is collected and used to instantiate the model
    """
    x,y,z = model.eyes_input_shape
    random_eye_array = np.random.rand(x, y, z)
    x,y,z = model.ears_input_shape
    random_ear_array = np.random.rand(x, y, z)

    eyes_time_sequence_observation = []
    ears_time_sequence_observation = []
    for _ in range(model.sequence_length):
        eyes_time_sequence_observation.append(random_eye_array)
        ears_time_sequence_observation.append(random_ear_array)

    state = (np.array([eyes_time_sequence_observation]), np.array([ears_time_sequence_observation]))

    return state

def save_model(model, file_path=_MODEL_FILE_LOCATION):
    if not file_path:
        file_path=_MODEL_FILE_LOCATION
    model.model_in_training.actor.save(file_path + "_actor")
    model.model_in_training.critic.save(file_path + "_critic")
    model.model_in_training.wonder.save(file_path + "_wonder")

def load_model(model, file_path):
    if not file_path:
        file_path=_MODEL_FILE_LOCATION
    if os.path.exists(file_path + "_actor"):
        model.model_in_training.actor=tf.keras.models.load_model(file_path + "_actor")
        model.model_in_training.critic=tf.keras.models.load_model(file_path + "_critic")
        model.model_in_training.wonder=tf.keras.models.load_model(file_path + "_wonder")

