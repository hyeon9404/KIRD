import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.svm import SVC
import argparse
import os

def preprocess(opt):
    trf = open('/home/sil/quantum/data_train_origin.csv', 'r', encoding='utf-8-sig')
    trl = open('/home/sil/quantum/label_train.csv', 'r', encoding='utf-8-sig')
    tef = open('/home/sil/quantum/data_test_origin.csv', 'r', encoding='utf-8-sig')
    tel = open('/home/sil/quantum/label_test.csv', 'r', encoding='utf-8-sig')
    seed = 1000
    np.random.seed(seed)
    train_features = np.loadtxt(trf, delimiter = ',')
    train_labels = np.loadtxt(trl, delimiter = ',')
    test_features = np.loadtxt(tef, delimiter = ',')
    test_labels = np.loadtxt(tel, delimiter = ',')

    if opt.normalize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)

    # 라벨별 데이터 인덱싱
    label_0_data = train_features[0:840]
    label_1_data = train_features[840:1680]
    label_2_data = train_features[1680:2520]

    # x/3 개의 데이터 선택
    num_each_label = opt.trainnum // 3  # 각 라벨당 선택될 데이터 수

    selected_indices_0 = np.random.choice(label_0_data.shape[0], num_each_label, replace=False)
    selected_label_0 = label_0_data[selected_indices_0]

    selected_indices_1 = np.random.choice(label_1_data.shape[0], num_each_label, replace=False)
    selected_label_1 = label_1_data[selected_indices_1]

    selected_indices_2 = np.random.choice(label_2_data.shape[0], num_each_label, replace=False)
    selected_label_2 = label_2_data[selected_indices_2]

    # 선택된 데이터를 합쳐 학습 세트 생성
    final_train_features = np.vstack((selected_label_0, selected_label_1, selected_label_2))
    final_train_labels = np.hstack((np.zeros(num_each_label), np.ones(num_each_label), np.full(num_each_label, 2)))

    # 학습 세트 셔플
    final_train_features, final_train_labels = shuffle(final_train_features, final_train_labels, random_state=seed)

    test_label_0_data = test_features[0:360]
    test_label_1_data = test_features[360:720]
    test_label_2_data = test_features[720:1080]

    num_each_label_test = opt.testnum // 3

    # print(test_label_0_data.shape[0])
    # print(num_each_label_test)
    selected_indices_0_test = np.random.choice(test_label_0_data.shape[0], num_each_label_test, replace=False)
    selected_label_0_test = test_label_0_data[selected_indices_0_test]

    selected_indices_1_test = np.random.choice(test_label_1_data.shape[0], num_each_label_test, replace=False)
    selected_label_1_test = test_label_1_data[selected_indices_1_test]

    selected_indices_2_test = np.random.choice(test_label_2_data.shape[0], num_each_label_test, replace=False)
    selected_label_2_test = test_label_2_data[selected_indices_2_test]

    final_test_features = np.vstack((selected_label_0_test, selected_label_1_test, selected_label_2_test))
    final_test_labels = np.hstack((np.zeros(num_each_label_test), np.ones(num_each_label_test), np.full(num_each_label_test, 2)))
    final_test_features, final_test_labels = shuffle(final_test_features, final_test_labels, random_state=seed)

    return final_train_features, final_train_labels, final_test_features, final_test_labels

def main(opt):

    train_features, train_labels, test_features, test_labels = preprocess(opt)

    path = f"/home/sil/quantum/outputs/{opt.instance}-{opt.kernel}-{opt.feature_map}-{opt.numqubits}/{opt.reps}-{opt.trainnum}-{opt.testnum}-{opt.normalize}"

    training_kernel_matrix = np.load(f'{path}/training_kernel_matrix.npy')
    training_time = np.load(f'{path}/training_time.npy')
    test_kernel_matrix = np.load(f'{path}/test_kernel_matrix.npy')
    test_time = np.load(f'{path}/test_time.npy')

    svc = SVC()
    svc.fit(train_features, train_labels)
    classic_score = svc.score(test_features, test_labels)

    svc = SVC(kernel="precomputed")
    svc.fit(training_kernel_matrix, train_labels)
    quantum_score = svc.score(test_kernel_matrix, test_labels)

    fig, ax = plt.subplots(figsize=(4+opt.trainnum//50, 4+opt.trainnum//50))
    ax.imshow(
        np.asmatrix(training_kernel_matrix), aspect="equal", interpolation="nearest", origin="upper", cmap="Blues"
    )
    ax.annotate(
        f'Computing time: {training_time/60:.2f}mins',
        xy = (1.0, -0.1),
        xycoords='axes fraction',
        ha='right',
        va="center",
    )
    fig.tight_layout()
    fig.savefig(fname=f'{path}/training_result.jpg')

    fig, ax = plt.subplots(figsize=(4+opt.trainnum//50, 4+opt.testnum//50))
    ax.imshow(
        np.asmatrix(test_kernel_matrix), aspect="equal", interpolation="nearest", origin="upper", cmap="Reds"
    )
    ax.annotate(
        f'Classic score: {classic_score:.2f}, Quantum score: {quantum_score:.2f},\nComputing time: {test_time/60:.2f}mins',
        xy = (1.0, -0.1),
        xycoords='axes fraction',
        ha='right',
        va="center",
    )
    fig.tight_layout()
    fig.savefig(fname=f'{path}/test_result.jpg')

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='CPU', help='')
    parser.add_argument('--instance', type=str, default='aer', help='aer, statevector, IBM backends')
    parser.add_argument('--kernel', type=str, default='', help='rbf_kernel, fidelity_kernel')
    parser.add_argument('--feature_map', type=str, default='', help='real_amp, zzfeature')
    parser.add_argument('--numqubits', type=int, default=4, help='hyperparameters path')
    parser.add_argument('--reps', type=int, default=2)
    parser.add_argument('--trainnum', type=int, default=100, help='initial weights path')
    parser.add_argument('--testnum', type=int, default=100)
    parser.add_argument('--normalize', action='store_true', help='normalize the data before splitting by label')
    # parser.add_argument('--batch-size', type=int, default=12, help='total batch size for all GPUs, -1 for autobatch')
    # parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    # parser.add_argument('--name', default='exp', help='save to project/name')
    # parser.add_argument('--runs', default='runs')
    
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)