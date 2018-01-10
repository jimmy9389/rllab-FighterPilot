import socket

BUFSIZE    = 512
ACTIONSIZE = 16
MAXWAITNUM = 20
'''
configFile = open('config/TCPServerInfo.cfg', 'r')
portInfo = configFile.readline()
infolist = portInfo.split()
if infolist[0] != 'PORT':
	print('Wrong config file!\n')
PORT = int(infolist[1])
'''

class SocketSever:

	def __init__(self,IP,PORT):
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
		self.sock.bind(('0.0.0.0', PORT))
		self.sock.listen(MAXWAITNUM)
		#self.sock.settimeout(5)
		self.conn = None

	def accept(self):
		self.conn, addr = self.sock.accept()
		#print(addr)

	def handleMessage(self):
		recvStateCnt = 0
		state1Param = []
		state2Param = []

		action = ['V','a','a','a','a','a','a','a','V','a','a','a','a','a','a','a']
	#	action[7] = action[15] = 'p'
		sfile = open('stateFile.txt', 'w')

		while 1:
			state = self.conn.recv(BUFSIZE)
			state = state.decode()
			recvStateCnt += 1

			timeOut = int(state[len(state)-1])
			missileRunOut = int(state[len(state)-3])
			if timeOut:
				sfile.write('----------TimeOut---------\n')
				break
			if missileRunOut:
				sfile.write('----------Missiles Run out---------\n')
				break

			print("ok")
			splitIndex = state.find('@')
			strState1 = state[:splitIndex]
			strState2 = state[splitIndex+1 : len(state)-4]
#			sfile.write('['+str(recvStateCnt)+']\n' + strState1 + '\n' + strState2 + '\n')

			strState1Param = strState1.split()
			strState2Param = strState2.split()


			#list[0:2]: the position of plane x,y,z 
			state1Param = [float(x) for x in strState1Param]
			state2Param = [float(x) for x in strState2Param]
			sfile.write('['+str(recvStateCnt)+']\n' + str(state1Param) + '\n' + str(state2Param) + '\n')
			print('['+str(recvStateCnt)+']\n' + str(state1Param) + '\n' + str(state2Param) + '\n')



			if state1Param[14] != 0.0:
				sfile.write('----------Plane-1 Die---------\n')
				break

			if state2Param[14] != 0.0:
				sfile.write('----------Plane-2 Die---------\n')
				break

			#ToDo: add RLLAB module to get action
			#Input: state1Param state2Param
			#Output: action

			#Define rule
			# if recvStateCnt <= 600:
			# 	action[3] = 'p'
			# elif recvStateCnt <= 1200:
			# 	action[3] = 'a'
			# 	action[7] = 'p'
			# elif recvStateCnt <= 1800:
			# 	action[7] = 'a'
			# 	action[15]= 'p'
			# 	if state2Param[9] < 3.1:
			# 		action[15] = 'a'
			# else:
			# 	if state1Param[4]%6.28 < 5.7:
			# 		action[4] = 'p'
			# 	else:
			# 		action[4] = 'a'

			# 	if state1Param[5] == 0.0 or state1Param[5]%6.29 > 6.0:
			# 		action[2] = 'p'
			# 	else:
			# 		action[2] = 'a' 

			self.conn.sendall(''.join(action).encode(encoding='utf-8'))

		sfile.close()

	def close(self):
		if not self.conn is None:
			self.conn.close()
		self.sock.close()
		



class SocketClient(object):
	def __init__(self):
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		
	
	def connect_sever(self,address,port):
		self.sock.connect((address,port))

	def close(self):
		self.sock.close()




	
