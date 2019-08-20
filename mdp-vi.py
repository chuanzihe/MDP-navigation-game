import numpy as np
import time
import copy
import operator

orientations = [(0,-1),(-1,0),(0,1),(1,0)]
start_time = time.time()


def print_curr_state_dir(curr_state_dir):
	# print "state_dir"
	# print curr_state_dir
	# arrows =  {(1, 0):'v', (0, 1):'>', (-1, 0):'^', (0, -1):'<', 'DST': '.'}
	# arrows = {(0,1):'v', (1,0):'>', (0,-1):'^', (-1,0):'<', 'DST': '.'}
	# arrows =  {(1, 0):'>', (0, 1):'v', (-1, 0):'<', (0, -1):'^', 'DST': '.'}

	arrows =  {(1, 0):u'\u2192 ', (0, 1):u'\u2193 ', (-1, 0):u'\u2190 ', (0, -1):u'\u2191 ', 'DST': 'E '}
	s = int(np.sqrt(len(curr_state_dir)))
	for j in range(0, s):
		str = ""
		for i in range(0, s):
			str += arrows[(curr_state_dir[(i,j)])]
		print str

def if_(test, result, alternative):
	"""(test ? result : alternative)
	"""
	if test:
		if callable(result): return result()
		return result
	else:
		if callable(alternative): return alternative()
		return alternative

def vector_add(a, b):
	"""Component-wise add
	waste time
	"""
	# import ipdb
	# ipdb.set_trace()
	return (a[0]+b[0],a[1]+b[1])
	# return tuple(map(operator.add, a, b))

def turn_left(orientation):
	return orientations[orientations.index(orientation)-1]

def turn_right(orientation):
	return orientations[(orientations.index(orientation)+1) % len(orientations)]

# # creat map for faster turn

# def turn_left(orientation):
# 	# [(0,-1),(-1,0),(0,1),(1,0)]
# 	dict = {
# 	(0,-1): (1,  0),
# 	(-1,0): (0, -1),
# 	(0, 1): (-1, 0),
# 	(1, 0): (0,  1)
# 	}
# 	return dict[orientation]

# def turn_right(orientation):
# 	# [(0,-1),(-1,0),(0,1),(1,0)]
# 	dict = {
# 	(0,-1): (-1,  0),
# 	(-1,0): (0,   1),
# 	(0, 1): (1, 0),
# 	(1, 0): (0,  -1)
# 	}
# 	return dict[orientation]

class MDP(object):
	"""docstring for Solution"""
	def __init__(self):
		super(MDP, self).__init__()
		self.read_input()
		self.eps = 0.1
		self.gamma = 0.9
		self.delta = np.array((self.eps*(1-self.gamma))/self.gamma, dtype='float64')
		self.actlist = orientations	
		# self.states = set()
		self.states = []
		for i in range(self.s_grid):
			for j in range(self.s_grid):				
				self.states.append((i,j))
		
		self.reward_no_car = {}
		for state in self.states:
			self.reward_no_car[state] = -1
		for state in self.loc_obs:
			self.reward_no_car[state] -= 100 #@note -100

		self.init_utils = self.reward_no_car.copy()

	def R(self, state):
		return self.reward_no_car[state]

	def T(self, state, action):
		"""
		(result-state, probability) pairs

		"""
		if action == 'None': # dest state no
			return [(0.0, state)]
		else:			
			return [(0.7, self.go(state, action)),
					(0.1, self.go(state, turn_right(action))),
					(0.1, self.go(state, turn_left(action))),
					(0.1, self.go(state, turn_left(turn_left(action))))]

	def actions(self,state):
		""" actions available"""
		if state in self.terminals:		
			return 'None' 
		else:
			return self.actlist

	def go(self, state, direction):
		"Return the state that results after take a step in direction dir."
		state1 = vector_add(state, direction)
		return if_(state1 in self.states, state1, state)

	def read_input(self):
		info = []
		dir = 'input.txt'
		with open(dir,'r') as f:
			for line in f.readlines():
				info.append(line.strip().split(','))
		self.s_grid =  int(info[0][0])
		self.num_car = int(info[1][0])
		self.num_obs = int(info[2][0])

		self.loc_obs = []
		for i in range(3,3+self.num_obs):
			self.loc_obs.append((int(info[i][0]), 
								 int(info[i][1])))
		self.loc_starts = []
		for i in range(3+self.num_obs, 3+self.num_obs+self.num_car):
			self.loc_starts.append((
							int(info[i][0]),
							int(info[i][1])))
		self.loc_ends = []
		for i in range(3+self.num_obs+self.num_car,
					   3+self.num_obs+self.num_car+self.num_car):
			self.loc_ends.append((
						  int(info[i][0]),
						  int(info[i][1])))

def main():

	mdp = MDP()
	def value_iteration(U1):
		R, T, gamma = mdp.R, mdp.T, mdp.gamma
		num_s = mdp.s_grid
		curr_state_dir = dict([(s, -1) for s in mdp.states])
		curr_state_u   = dict([(s, -1) for s in mdp.states])
		U = copy.deepcopy(U1)
		while True:
			delta = 0.
			for s in mdp.states:
				# U1[s] = R(s) + gamma * max([sum([p * U[s1] for (p, s1) in T(s, a)])
				# 							for a in mdp.actions(s)])
				if U[s] == 99: # reached destination
					U[s] = 99
					curr_state_u[s]   = 99
					curr_state_dir[s] = 'DST'
					continue
				max_u = -999999 #@todo
				dir   = -1
				
				# t0 = time.time()		
				for a in mdp.actions(s):
				# for a in mdp.actlist:
					u = U1[s]

					# clean but slower version
					# for (p, s1) in T(s,a):
					# 	u += gamma * p * U[s1]

					for turn in mdp.actions(s):
						s1 = (s[0]+turn[0], s[1]+turn[1])
						if a == turn:
							# if s1 not in mdp.states:
							if s1[1]<0 or s1[0]<0 or s1[1]>=num_s or s1[0]>=num_s:
								u += gamma * 0.7 * U[s]
							else:
								u += gamma * 0.7 * U[s1]
						else:
							# if s1 not in mdp.states:
							if s1[1]<0 or s1[0]<0 or s1[1]>=num_s or s1[0]>=num_s:
								u += gamma * 0.1 * U[s]
							else:
								u += gamma * 0.1 * U[s1]

					if u > max_u:
						max_u = u
						dir   = a

				curr_state_u[s]	= max_u
				curr_state_dir[s]  = dir
				# delta = max(delta, abs(U1[s] - U[s])) #cur state u: U[s], prev state U1
				delta = max(delta, abs(curr_state_u[s] - U[s])) #cur state u: U[s], prev state U1
				# t1 = time.time()
				# print(t1-t0)
			# if delta < mdp.eps * (1 - gamma) / gamma:
			if delta < mdp.delta:
				# return U
				break
			else:
				U = copy.deepcopy(curr_state_u)
		# print_curr_state_dir(curr_state_dir)
		return curr_state_dir

	results = []
	for car in range(mdp.num_car):
		# if car == 2:
		# 	import ipdb
		# 	ipdb.set_trace()		
		start = mdp.loc_starts[car]
		end = mdp.loc_ends[car]
		util = copy.deepcopy(mdp.reward_no_car)
		util[end] += 100
		mdp.terminals = end


		cur_dirs = value_iteration(util)
		sum_total = 0
		for seed_i in range(10):
			np.random.seed(seed_i)
			swerve = np.random.random_sample(1000000) 
			k=0 
			sum=0
			pos = copy.deepcopy(start)
			while pos != end:
				move = cur_dirs[pos] 
				if swerve[k] > 0.7:
					if swerve[k] > 0.8:
						if swerve[k] > 0.9:
							move = turn_left(turn_left(move)) 
						else:
							move = turn_left(move) 
					else:
						move = turn_right(move)
				pos = mdp.go(pos,move)
				sum += util[pos]
				k+=1
				# print('%d: %d'%(k,sum))
			# print(sum)
			sum_total += sum
		result = int(np.floor(float(sum_total)/10))
		results.append(result)
		# print(sum_total)
		# print('result: %d'%(result))

	with open('output.txt','w') as f:
		for result in results:
			f.write(str(result)+'\n')

if __name__ == '__main__':
	main()
	# print "\ntime: "+ str(time.time() - start_time)