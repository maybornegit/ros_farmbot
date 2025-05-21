from .nicolet_model_helpers import residual_function_kva, basic_test_w_lighting
import numpy as np

def optimize_day(traj, light, params=[4.0e-7, 22.1, 0.5]):
    day_range = np.arange(-1,6,4/12)
    print('---------------- Data Loading ----------------')
    res_ = []
    new_traj = [traj[0][:],traj[1][:]]
    for day in day_range:
        dt = 1
        traj_ = [[np.array([t-day for t in new_traj[0]]), new_traj[1]]]
        arg = [traj_, [light], dt, False, False]
        resid, _ = residual_function_kva(params, arg)
        res_.append(resid)
    best_day_idx = min(enumerate(res_), key=lambda x: x[1])[0]
    traj = [[t-day_range[best_day_idx] for t in traj[0]], traj[1]]
    print(day_range[best_day_idx])
    return traj

def pull_wgt_per_area(raw_trajs):##### Find trajectories divided by area
    factors = [3.5, 5]
    cutoffs = [5,26]
    trajs = []
    in_conv = 0.0254
    for i in range(len(raw_trajs)):
        cur_traj = np.array(raw_trajs[i],dtype=np.float64)
        for j in range(cur_traj.shape[1]):
            if cur_traj[1,j] <= cutoffs[0]:
                cur_traj[1,j] /= (factors[0]*4.5*in_conv**2)*1000
            elif cur_traj[1,j] >= cutoffs[1]:
                cur_traj[1,j] /= (factors[1]*4.5*in_conv**2)*1000
            else:
                factor = (0.9+np.sqrt(0.69*cur_traj[1,j]))
                cur_traj[1,j] /= (factor*4.5*in_conv**2)*1000
        trajs.append(cur_traj)
    return trajs

def inverse_wgt_per_area(trajs_per_area):##### Find trajectories divided by area
    factors = [3.5,2.5,3.0,3.5,4.0,4.5,5]
    cutoffs_mass = [3,5,8,11,16,22,26]
    trajs = []
    in_conv = 0.0254
    cutoffs = [c/(factors[k]*4.5*in_conv**2)/1000 for k, c in enumerate(cutoffs_mass)]
    for i in range(len(trajs_per_area)):
        cur_traj = np.array(trajs_per_area[i],dtype=np.float64)
        for j in range(cur_traj.shape[1]):
            if cur_traj[1, j] - cutoffs[0] <= 1e-4:
                cur_traj[1,j] *= (factors[0]*4.5*in_conv**2)*1000
            elif cur_traj[1, j] >= cutoffs[-1]:
                cur_traj[1,j] *= (factors[-1]*4.5*in_conv**2)*1000
            else:
                b = -cur_traj[1,j]*4.5*in_conv**2*1000*np.sqrt(0.69)
                c = -cur_traj[1,j]*4.5*in_conv**2*1000*0.9
                x = ((-b+np.sqrt(b**2-4*c))/(2))**2
                cur_traj[1,j] = x
        trajs.append(cur_traj)
    return trajs

def projected_mass(cur_meas, light, day_predictions, params=[4.0e-7, 22.1, 0.5]):
    trajs = pull_wgt_per_area([cur_meas])
    traj = optimize_day(trajs[0], light, params=params)
    # trajs = inverse_wgt_per_area([traj])
    cur_day = traj[0][-1]

    prediction_dates = [cur_day+i for i in day_predictions]
    t_traj, _, y_traj = basic_test_w_lighting(light, params[0], params[1], params[2], 1)
    predicted_wgts = [[],[]]
    for prdt in prediction_dates:
        for k,t in enumerate(t_traj):
            if np.abs(t/60/60/24 - prdt) <= 1e-6:
                predicted_wgts[0].append(t/60/60/24)
                predicted_wgts[1].append(y_traj[1][k])
                break
    trajs = inverse_wgt_per_area([predicted_wgts])
    return trajs[0][1]

if __name__ == '__main__':
    ### Fake trajectory
    init_traj = [[],[]]
    for i in range(3):
        init_traj[0] += [7+i+(j/24) for j in range(14)]
        mass = [np.exp(0.25*(i+(j/24))) for j in range(14)]
        noise = np.random.normal(0,0.2,14)
        init_traj[1] += [mass[j] + noise[j] for j in range(len(mass))]

    ### Need a way to pull init_traj + pull light for a position (current_pos // 3)
    masses = projected_mass(init_traj, 300.0, [0,1,3,5,7,10], params=[2.34e-7, 22.1,0.5])

