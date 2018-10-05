import numpy as np
import os
import ipdb


def connect_dataset(dist_start, dist_end, file_list, outputdir,
                    topodx=5, offset=5000, gclass_num=4, test_data_num=100):
    """
    複数のデータセットを連結する
    """
    # ipdb.set_trace()

    # 学習領域の始点と終点を決めて，グリッド番号に変換する
    prox = np.round((dist_start+offset)/topodx).astype(np.int32)
    dist = np.round((dist_end+offset)/topodx).astype(np.int32)
    H = np.zeros([0, (dist-prox) * (gclass_num)])
    icond = np.zeros([0, gclass_num + 3])

    # ファイルの読み込みと結合
    for i in range(len(file_list)):
        H_temp = np.loadtxt(
            file_list[i] + '/H1.txt', delimiter=',')[:, prox:dist]
        for j in range(gclass_num - 1):
            H_next = np.loadtxt(
                file_list[i] + '/H{}.txt'.format(j + 1), delimiter=',')[:, prox:dist]
            H_temp = np.concatenate([H_temp, H_next], axis=1)
        icond_temp = np.loadtxt(
            file_list[i] + '/initial_conditions.txt', delimiter=',')
        if icond_temp.shape[0] != H_temp.shape[0]:
            icond_temp = icond_temp[:-1, :]
        H = np.concatenate((H, H_temp), axis=0)
        icond = np.concatenate((icond, icond_temp), axis=0)

    # データの最大値と最小値を取得する
    max_x = np.max(H)
    min_x = np.min(H)
    icond_max = np.max(icond, axis=0)
    icond_min = np.min(icond, axis=0)

    # データをテストとトレーニングに分割する
    H_train = H[0:-test_data_num, :]
    H_test = H[H.shape[0] - test_data_num:, :]
    icond_train = icond[0:-test_data_num, :]
    icond_test = icond[H.shape[0] - test_data_num:, :]

    # データを保存する
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    np.savetxt(outputdir + '/H_train.txt', H_train, delimiter=',')
    np.savetxt(outputdir + '/H_test.txt', H_test, delimiter=',')
    np.savetxt(outputdir + '/icond_train.txt', icond_train, delimiter=',')
    np.savetxt(outputdir + '/icond_test.txt', icond_test, delimiter=',')
    np.savetxt(outputdir + '/icond_min.txt', icond_min, delimiter=',')
    np.savetxt(outputdir + '/icond_max.txt', icond_max, delimiter=',')
    np.savetxt(outputdir + '/x_minmax.txt', [min_x, max_x], delimiter=',')


if __name__ == "__main__":
    dist_start = 0
    dist_end = 900
    original_data_dir = "Y:/naruse/TC_training_data_G4"
    parent_dir = "Z:/Documents/PythonScripts/DeepLearningTurbidite/20181005/G4"
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
    outputdir = parent_dir + "/data"
    # file_list = []
    # for i in range(1,17):
    #     file_list.append(original_data_dir + "/TCModel_for_ML{0:02d}/output".format(i))
    # del file_list[2]
    file_list = [original_data_dir + "/TCModel_for_ML_G4/output"]
    connect_dataset(dist_start, dist_end, file_list, outputdir,
                    offset=50, gclass_num=4, test_data_num=200)
