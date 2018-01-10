from sys import argv
from rllab.envs.socketManager import SocketSever,SocketClient
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
#from rllab.baselines.deterministic_mlp_baseline import DeterministicMLPBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.fighterpilot_env import FighterpilotEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.box2d.cartpole_env import CartpoleEnv 
#import tensorflow as tf
import rllab.misc.global_params as gp
from rllab.misc.instrument import run_experiment_lite


'''if argv[1] == None:
    print("please input params address")
if argv[2] == None:
    print("please input params PORT")

#print(type(int(argv[1])),int(argv[1]))
address = str(argv[1])
PORT = int(argv[2])'''
def run_task(*_):
    gp.sock_with_simulator =SocketSever('0.0.0.0',10011)
    #sock_with_param_sever = SocketClient()
    #参数统一到这个端口更新和读取
    #print("run sucess==========================================")
    #sock_with_param_sever.connect_sever(address,9998) #9998参数服务器的端口

    env = TfEnv(normalize(FighterpilotEnv('127.0.0.1',10011)))
    #env = TfEnv(normalize(CartpoleEnv()))
    #policy = GaussianMLPPolicy(
    policy = CategoricalMLPPolicy(
        name='policy',
        env_spec=env.spec,
        hidden_sizes=(128, 128),
        #hidden_nonlinearity=tf.nn.relu,
    ) 
    baseline = GaussianMLPBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        #socket=sock_with_param_sever,
        batch_size=600,
        max_path_length=15,
        n_itr=10000,
        discount=1,
        step_size=0.01,
        center_adv=False,
    )

    algo.train()
run_experiment_lite(
    run_task,
    exp_name="flight-epsilon-ma",
    n_parallel=1,
    snapshot_mode="last",
    seed=11,
    # plot=True
)
