from avatarpy import Avatar, dataset
csv_path = dataset['freely_moving'] # dataset is dict of csv_path provided by avatarpy package
avatar = Avatar(csv_path)
# avatar.animate(avatar.index[0:100]).save('freely_moving_0_to_100.html')
print(avatar.get_rolling_corr(avatar.velocity))
print(avatar.corr('velocity'))
print(avatar.corr('velocity', window=20))