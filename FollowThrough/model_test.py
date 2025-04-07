from UserModel import *
import os

def main():
    user_dict = {}
    files_added = []

    if os.path.isfile('test_files/pro_users_seen.pickle'):
        with open('test_files/pro_users_seen.pickle', 'rb') as f:
            files_added = pickle.load(f) 
            # print(f)

    if os.path.isfile('test_files/pro_user_dict.pickle'):
        with open('test_files/pro_user_dict.pickle', 'rb') as f:
            user_dict = pickle.load(f) 
    # files_added = []
    # user_dict={}
    for f in glob.iglob('./input_mp4s/test2k/*'):
        if f in files_added:
            print("already added ", f)
            continue

        print(f)
        files_added.append(f)

        athlete = f[:-5] + f[-4:]
        print(athlete)
        if athlete in user_dict:
            u = user_dict[athlete]
            u.add_sample(f)
        else:
            print('here')
            u = UserModel()
            ath_dict = u.add_sample(f)
            user_dict[athlete] = ath_dict # u
        print(athlete, u.stat_dict)
        with open('./test_files/pro_user_dict.pickle', 'wb') as handle:
            pickle.dump(user_dict, handle)

        with open('./test_files/pro_users_seen.pickle', 'wb') as handle:
            pickle.dump(files_added, handle)
    # print(user_dict["./input_mp4s/2kvids\\bird.mp4"].get_dict())
    print(user_dict)
    compare(user_dict)
    
def compare(user_dict):
    for athlete in user_dict:
        for pro in user_dict:
            if athlete!=pro:
                print("------------------------------------------")
                print (athlete, "VS", pro)
                print(similarity(user_dict[athlete], user_dict[pro]))

def similarity(athlete, pro):
    score = 0
    for stat in athlete:
        score+=abs(athlete[stat]-pro[stat])
    return score
if __name__ == '__main__':
    main()
