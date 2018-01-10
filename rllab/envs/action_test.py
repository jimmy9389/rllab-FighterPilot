action_sequence = ['V','a','a','a','a','a','a','a','V','a','a','a','a','a','a','a']
for action in range(0,9):
    # if_accelarate = int(action / 9)
    # if if_accelarate == 0:
    #     action_sequence[5]='p'
    # if if_accelarate == 1:
    #     action_sequence[6]='p'
    # spin_or_climb = action % 9
    # if (spin_or_climb%3 <= 1):
    #     action_sequence[action%3+1] ='p'
    # if_spin = int(spin_or_climb / 3)
    # if if_spin ==0:
    #     action_sequence[3] = 'p'
    # if if_spin ==1:
    #     action_sequence[4] = 'p'
    if (action%3) <= 1:
       action_sequence[action%3+1] = 'p'
    if_spin = int(action / 3)
    if if_spin == 0:
        action_sequence[3]='p'
    if if_spin == 1:
        action_sequence[4]='p'
    print(action_sequence[0:7])
    action_sequence = ['V','a','a','a','a','a','a','a','V','a','a','a','a','a','a','a']
