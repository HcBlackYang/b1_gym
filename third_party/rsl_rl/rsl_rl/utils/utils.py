# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch

def split_and_pad_trajectories(tensor, dones):
    """ 
    Splits trajectories at done indices, then pads them with zeros up to the length of the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories.
    
    Assumes input shape: [time, num_envs, feature_dim]
    """
    dones = dones.clone()
    dones[-1] = 1  # Force termination at the last step to avoid incomplete trajectories
    
    # Transpose to have shape (num_envs, time, feature_dim)
    tensor = tensor.transpose(1, 0)
    dones = dones.transpose(1, 0)
    
    # Flatten across environments
    flat_dones = dones.reshape(-1)
    
    # Get indices where episodes terminate
    done_indices = torch.cat((torch.tensor([-1], device=tensor.device), flat_dones.nonzero().squeeze()))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    
    # Extract trajectories
    trajectories = torch.split(tensor.reshape(-1, tensor.shape[-1]), trajectory_lengths.tolist())
    
    # Pad sequences
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories, batch_first=True)
    
    # Create mask
    max_len = padded_trajectories.shape[1]
    trajectory_masks = torch.arange(max_len, device=tensor.device).unsqueeze(0) < trajectory_lengths.unsqueeze(1)
    
    return padded_trajectories, trajectory_masks

def unpad_trajectories(padded_trajectories, masks):
    """
    Reconstructs the original unpadded trajectories from padded representations.
    """
    unpadded = padded_trajectories[masks]
    return unpadded.view(-1, padded_trajectories.shape[-1])