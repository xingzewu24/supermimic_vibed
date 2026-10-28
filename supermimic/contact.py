"""Last-physics-substep contact diagnostics; these values never enter rewards or termination."""

import torch
import warp as wp


class ContactDiagnostics:
    def __init__(self, env):
        self.env = env
        model, device = env.sim.mj_model, env.device
        ngeom = model.ngeom
        self.object_geom = torch.zeros(ngeom, dtype=torch.bool, device=device)
        self.object_geom[env.object.indexing.geom_ids] = True
        self.robot_body = torch.full((ngeom,), -1, dtype=torch.long, device=device)
        self.label_body = torch.full_like(self.robot_body, -1)
        for gid in env.robot.indexing.geom_ids.cpu().tolist():
            name = model.body(int(model.geom_bodyid[gid])).name.removeprefix("robot/")
            self.robot_body[gid] = env.robot.body_names.index(name)
            # Fixed links may be compiled separately; labels belong to the nearest named ancestor.
            bid = int(model.geom_bodyid[gid])
            while name not in env.motion.body_names and bid > 0:
                bid = int(model.body_parentid[bid])
                name = model.body(bid).name.removeprefix("robot/")
            if name in env.motion.body_names:
                self.label_body[gid] = env.motion.body_names.index(name)
        self.hand_labels = torch.tensor([n.startswith(("L_", "R_")) for n in env.motion.body_names], device=device)

    @torch.no_grad()
    def compute(self, frames):
        env = self.env
        data = env.sim.wp_data
        geom = wp.to_torch(data.contact.geom).long()
        world = wp.to_torch(data.contact.worldid).long()
        distance = wp.to_torch(data.contact.dist)
        active = torch.arange(len(world), device=env.device) < wp.to_torch(data.nacon)[0]
        active &= (world >= 0) & (world < env.N) & (geom >= 0).all(-1) & (distance <= 0)
        # Unused contact slots are never indexed until their validity has been checked.
        geom, world, distance = geom[active], world[active], distance[active]
        object_pair = self.object_geom[geom[:, 0]] ^ self.object_geom[geom[:, 1]]
        robot_geom = torch.where(self.object_geom[geom[:, 0]], geom[:, 1], geom[:, 0])
        use = object_pair & (self.label_body[robot_geom] >= 0)
        geom_r, world, distance = robot_geom[use], world[use], distance[use]
        labels = self.label_body[geom_r]
        touched = torch.zeros(env.N, len(env.motion.body_names), device=env.device, dtype=torch.bool)
        touched[world, labels] = True
        target = env.motion.contact_label[frames]
        valid = target != 0
        matched = ((touched == (target == 1)) & valid).sum(-1) / valid.sum(-1).clamp_min(1)
        penetration = torch.zeros(env.N, device=env.device)
        penetration.scatter_reduce_(0, world, -distance, reduce="amax", include_self=True)
        # Relative tangential velocity at each physical contact point.
        pos = wp.to_torch(data.contact.pos)[active][use]
        normal = wp.to_torch(data.contact.frame)[active][use, 0]
        rb = self.robot_body[geom_r]
        robot_vel = env.robot.data.body_link_vel_w[world, rb]
        robot_pos = env.robot.data.body_link_pos_w[world, rb]
        object_vel = env.object.data.root_link_vel_w[world]
        object_pos = env.object.data.root_link_pos_w[world]
        rv = robot_vel[:, :3] + torch.cross(robot_vel[:, 3:], pos - robot_pos, dim=-1)
        ov = object_vel[:, :3] + torch.cross(object_vel[:, 3:], pos - object_pos, dim=-1)
        relative = rv - ov
        tangential = relative - (relative * normal).sum(-1, keepdim=True) * normal
        slip, count = torch.zeros_like(penetration), torch.zeros_like(penetration)
        slip.scatter_add_(0, world, tangential.norm(dim=-1))
        count.scatter_add_(0, world, torch.ones_like(distance))
        needs_grasp = ((target == 1) & self.hand_labels).any(-1)
        has_grasp = (touched & self.hand_labels).any(-1)
        return {"contact_match_rate": matched, "object_penetration": penetration,
                "contact_slip_speed": slip / count.clamp_min(1),
                "grasp_loss": (needs_grasp & ~has_grasp).float()}
