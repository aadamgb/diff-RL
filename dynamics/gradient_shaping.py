import torch

class CustomGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, action, model_inst, control_mode):
        # Full dynamics for the actual trajectory
        with torch.no_grad():
            next_state = model_inst.full_dynamics(state, action, control_mode)
        
        # Save for backward pass
        ctx.save_for_backward(state, action)
        ctx.model_inst = model_inst
        ctx.control_mode = control_mode
        
        return next_state

    @staticmethod
    def backward(ctx, grad_output):
        state, action = ctx.saved_tensors
        model = ctx.model_inst
        
        state_vars = state.detach().requires_grad_(True)
        action_vars = action.detach().requires_grad_(True)
        
        with torch.enable_grad():
            # Slice state: simplified_dynamics only take [x, y, vx, vy, theta, omega]
            state_simplified = state_vars[..., :6]
            
            # Run the simplified dynamics
            next_state_simple = model.simplified_dynamics(
                state_simplified, action_vars, ctx.control_mode
            )
            
            # Chain Rule: dLoss/dInput = dLoss/dNextState * dNextState/dInput
            # We use grad_output[..., :6] because simplified model only has 6 states
            grads = torch.autograd.grad(
                next_state_simple, 
                (state_vars, action_vars), 
                grad_outputs=grad_output[..., :6],
                allow_unused=True
            )
            
        # Return grads for (state, action, model_inst, control_mode)
        # model_inst and control_mode get None because they aren't differentiable
        return grads[0], grads[1], None, None