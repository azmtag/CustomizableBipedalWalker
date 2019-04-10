import numpy as np
import warnings
from gym.envs.box2d.bipedal_walker import *
class CustomizableBipedalWalker(BipedalWalker):
    def __init__(self):
        self.default_params = {
            'stump_height_low': 1,
            'stump_height_high': 3,
            'pit_depth': 4,
            'pit_width_low': 3,
            'pit_width_high': 5,
            'stair_heights': [-.5, .5],
            'stair_width_low': 4,
            'stair_width_high': 5,
            'stair_steps_low': 3,
            'stair_steps_high': 5,
            'states': [0],
            'state_probs': None
        }
        self.params = {**self.default_params}
        BipedalWalker.__init__(self)
        
    def _update_env_params(self, **kwargs):
        # TODO: add kind of sanity check here
        self.params = {**self.params, **kwargs}
        _ = self.reset()
        
    def reset_env_params(self, hardcore=False):
        params = {**self.default_params}
        if hardcore:
            params['states'] = np.arange(4)
        self._update_env_params(**params)
    
    def set_env_states(self, state_mask, p=None):
        """
        :param state_mask: np.array(,dtype=bool) that masks ["GRASS", "STUMP", "STAIRS", "PIT"].
            Note that masking out "GRASS" takes no effect.
        :param p: np.array or list of probabilities: [p_grass, p_stump, p_stairs, p_pit].
            Probs corresponding to masked out states are ignored
        :return: None
        """
        states_ = np.arange(4)[state_mask]
        p_ = None
        if p is not None:
            p_ = np.array(p)
            if not np.all(p_ >= 0):
                raise ValueError
            p_ = p_[state_mask] / p_[state_mask].sum()
        self._update_env_params(states=states_, state_probs=p_)
    
    def set_env_params(self, pit_width=None, stair_width=None, stair_steps=None, stump_height=None):
        """
            NB: All params are integers or tuples of integers
        """
        kwargs = {**locals()}
        _ = kwargs.pop('self', None)
        params = {}
        for k,v in kwargs.items():
            if type(v) is int:
                params[k + '_low'] = v
                params[k + '_high'] = v + 1
            elif isinstance(v, (tuple, list)): 
                if v[1] - v[0] >= 1:
                    params[k + '_low'] = v[0]
                    params[k + '_high'] = v[1]
                else:
                    warnings.warn(f'{k} shoud be an integer. {k}[1] - {k}[0] < 1 '+\
                                  f'=> will set {k}_low = {v[0]}, {k}_high = {v[0]+1}')
                    params[k + '_low'] = v[0]
                    params[k + '_high'] = v[0] + 1
        self._update_env_params(**params)
        
    def _generate_terrain(self, hardcore=True):
        GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
        state    = GRASS
        velocity = 0.0
        y        = TERRAIN_HEIGHT
        counter  = TERRAIN_STARTPAD
        oneshot  = False
        self.terrain   = []
        self.terrain_x = []
        self.terrain_y = []
        for i in range(TERRAIN_LENGTH):
            x = i*TERRAIN_STEP
            self.terrain_x.append(x)

            if state==GRASS and not oneshot:
                velocity = 0.8*velocity + 0.01*np.sign(TERRAIN_HEIGHT - y)
                if i > TERRAIN_STARTPAD: velocity += self.np_random.uniform(-1, 1)/SCALE   #1
                y += velocity

            elif state==PIT and oneshot:
                counter = self.np_random.randint(self.params['pit_width_low'], 
                                                 self.params['pit_width_high'])
                PIT_H = self.params['pit_depth']
                poly = [
                    (x,              y),
                    (x+TERRAIN_STEP, y),
                    (x+TERRAIN_STEP, y-PIT_H*TERRAIN_STEP),
                    (x,              y-PIT_H*TERRAIN_STEP),
                    ]
                self.fd_polygon.shape.vertices=poly
                t = self.world.CreateStaticBody(
                    fixtures = self.fd_polygon)
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)

                self.fd_polygon.shape.vertices=[(p[0]+TERRAIN_STEP*counter,p[1]) for p in poly]
                t = self.world.CreateStaticBody(
                    fixtures = self.fd_polygon)
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)
                counter += 2
                original_y = y

            elif state==PIT and not oneshot:
                y = original_y
                if counter > 1:
                    y -= PIT_H*TERRAIN_STEP

            elif state==STUMP and oneshot:
                counter = self.np_random.randint(self.params['stump_height_low'], self.params['stump_height_high'])
                poly = [
                    (x,                      y),
                    (x+counter*TERRAIN_STEP, y),
                    (x+counter*TERRAIN_STEP, y+counter*TERRAIN_STEP),
                    (x,                      y+counter*TERRAIN_STEP),
                    ]
                self.fd_polygon.shape.vertices=poly
                t = self.world.CreateStaticBody(
                    fixtures = self.fd_polygon)
                t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                self.terrain.append(t)

            elif state==STAIRS and oneshot:
                stair_height = self.np_random.choice(self.params['stair_heights'])
                stair_width = self.np_random.randint(self.params['stair_width_low'], 
                                                     self.params['stair_width_high'])
                stair_steps = self.np_random.randint(self.params['stair_steps_low'], 
                                                     self.params['stair_steps_high'])
                original_y = y
                for s in range(stair_steps):
                    poly = [
                        (x+(    s*stair_width)*TERRAIN_STEP, y+(   s*stair_height)*TERRAIN_STEP),
                        (x+((1+s)*stair_width)*TERRAIN_STEP, y+(   s*stair_height)*TERRAIN_STEP),
                        (x+((1+s)*stair_width)*TERRAIN_STEP, y+(-1+s*stair_height)*TERRAIN_STEP),
                        (x+(    s*stair_width)*TERRAIN_STEP, y+(-1+s*stair_height)*TERRAIN_STEP),
                        ]
                    self.fd_polygon.shape.vertices=poly
                    t = self.world.CreateStaticBody(
                        fixtures = self.fd_polygon)
                    t.color1, t.color2 = (1,1,1), (0.6,0.6,0.6)
                    self.terrain.append(t)
                counter = stair_steps*stair_width

            elif state==STAIRS and not oneshot:
                s = stair_steps*stair_width - counter - stair_height
                n = s/stair_width
                y = original_y + (n*stair_height)*TERRAIN_STEP

            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter==0:
                counter = self.np_random.randint(TERRAIN_GRASS/2, TERRAIN_GRASS)
                if state==GRASS:
                    state = self.np_random.choice(self.params['states'], p=self.params['state_probs'])
                    oneshot = True
                else:
                    state = GRASS
                    oneshot = True

        self.terrain_poly = []
        for i in range(TERRAIN_LENGTH-1):
            poly = [
                (self.terrain_x[i],   self.terrain_y[i]),
                (self.terrain_x[i+1], self.terrain_y[i+1])
                ]
            self.fd_edge.shape.vertices=poly
            t = self.world.CreateStaticBody(
                fixtures = self.fd_edge)
            color = (0.3, 1.0 if i%2==0 else 0.8, 0.3)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.4, 0.6, 0.3)
            poly += [ (poly[1][0], 0), (poly[0][0], 0) ]
            self.terrain_poly.append( (poly, color) )
        self.terrain.reverse()
