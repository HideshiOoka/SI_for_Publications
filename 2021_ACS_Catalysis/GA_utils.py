import numpy as np
from HER_MK import get_diff, get_log_diff
import random
def crossover(parents, alpha_lock):
    children = np.ones(shape = parents.shape)
    if alpha_lock != 0:
        for idx in range(parents.shape[1]-4):
            children[:,idx] = random.choices(parents[:,idx], weights = 1/10**parents[:,-1], k = parents.shape[0])
        children[:,-4:-1] = alpha_lock    
    else:
        for idx in range(parents.shape[1]-1):
            children[:,idx] = random.choices(parents[:,idx], weights = 1/10**parents[:,-1], k = parents.shape[0])
    return children

def evolve(parents, alpha_lock):
    unique, count = np.unique(parents, axis=0, return_counts=True)
    duplicates = unique[count > 1]
    for duplicate in duplicates:
        repeated_idx = np.argwhere(np.all(parents == duplicate, axis = 1))
        evol_time = len(repeated_idx)
        evolved_parent = GD(duplicate, evol_time, alpha_lock)
        parents[repeated_idx[0,0]] = evolved_parent
    return parents   

def GD(parent, evol_time, alpha_lock):
    num_pars = len(parent)-1
    delta = 1E-5
    GD_cycles = 10 * evol_time
    old_par_vec = parent[0:-1] # actually has log_diff at the end
    old_diff = get_diff(old_par_vec)
    grad_vec = np.zeros(num_pars)
    
    if alpha_lock !=0:
        num_free_pars = num_pars -3
    else:
        num_free_pars = num_pars        
    for cycles in range(GD_cycles):
        for idx in range(num_free_pars):
            delta_vec = np.zeros(num_pars)
            delta_vec[idx] = delta
            # print(delta_vec[idx])
            grad_vec[idx] =(get_diff(old_par_vec+delta_vec) - old_diff)/delta

        new_par_vec = old_par_vec - 0.01 * grad_vec
        new_diff = get_diff(new_par_vec)
        if new_diff < old_diff:
            old_par_vec = new_par_vec
            old_diff = new_diff
        else:
            delta = delta/10
    old_log_diff = np.log10(old_diff)
    return np.append(old_par_vec, old_log_diff)

def save_pars(pop_size, num_generations = 50, num_pars = 6, num_trials = 100, mechanism = "VHT", alpha_lock = 0):
    results_table = np.zeros((num_trials,num_pars+1))   
    num_parents = int(pop_size * 0.5)
    num_children = pop_size - num_parents 
    for trial in range(num_trials):
        print(trial)
        pop = np.random.uniform(-0.95, 0.95, (pop_size,num_pars + 1)) #pars +  diff
        # change the scaling of each parameter
        pop[:,1] = pop[:,1] * 10 # -10 < logK2 <10
        pop[:,2] = pop[:,2] * 10 # -10 < logK2 <10
        pop[:,3:6] = 0.5 + pop[:,3:6]/2 # 0 < alpha <1
        if alpha_lock !=0: # Set alpha_lock to zero if alpha should be a free variable
            pop[:,3:6] = alpha_lock
        for p in range(pop_size):
            pop[p,-1] = get_log_diff(pop[p,0:-1])
        pop = pop[np.argsort(pop[:,-1])]
        pop_init = pop
        pop_dict= np.zeros((num_generations,pop_size,num_pars+1))
        for g in range(num_generations):
            pop_dict[g] = pop    
            parents = pop[0:num_parents]
            children = crossover(parents, alpha_lock)
            evolved_parents = evolve(parents, alpha_lock)
            pop = np.vstack((evolved_parents,children))
            for p in range(pop_size):
                pop[p,-1] = get_log_diff(pop[p,0:-1])
            pop = pop[np.argsort(pop[:,-1])]
            if pop[0,-1] == pop[-1,-1]:
                print("Optimization Converged at Generation : " + str(g))
                break
        results_table[trial] = pop[0]
    #
    np.savetxt(f"fitting_results/results_table_{mechanism}_{pop_size}_{alpha_lock}.csv", results_table, delimiter=',', header="GH,logK2,logK3,a1,a2,a3,log diff",comments='')    
    np.savetxt(f"fitting_results/pop_dict_{mechanism}_{pop_size}_{alpha_lock}.csv", pop_dict.reshape(pop_size*num_generations,num_pars+1), delimiter=',')    

