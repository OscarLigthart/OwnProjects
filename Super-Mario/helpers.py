import torch
import torch.nn.functional as F

def compute_q_val(model, state, action):
    # get q-values
    actions = model(state)
    q_val = torch.gather(actions, 1, action.view(-1, 1))

    return q_val


def compute_target(model, reward, next_state, done, discount_factor):
    # done is a boolean (vector) that indicates if next_state is terminal (episode is done)

    # calculate q-values and pick highest
    actions = model(next_state)
    _, chosen_action = actions.max(1)
    chosen_action = torch.gather(actions, 1, chosen_action.view(-1, 1))

    #  calculate target
    target = reward.view(chosen_action.shape) + (discount_factor * chosen_action)

    # set target to just the reward if next_state is terminal
    target[done] = reward[done].view(target[done].shape)

    return target

def train(model, memory, optimizer, batch_size, discount_factor):
    # DO NOT MODIFY THIS FUNCTION

    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    #print(state[0].shape)
    state = torch.stack(state).view(-1,3,60,64)
    next_state = torch.stack(next_state).view(-1,3,60,64)

    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)
    done = torch.tensor(done, dtype=torch.uint8)  # Boolean

    # compute the q value
    q_val = compute_q_val(model, state, action)

    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        target = compute_target(model, reward, next_state, done, discount_factor)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())