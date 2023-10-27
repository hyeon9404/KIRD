import numpy as np
import time

# Import Qiskit
from qiskit_ibm_runtime import QiskitRuntimeService, Options
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit_aer import AerError
from qiskit.compiler import transpile
from qiskit.tools.visualization import plot_histogram, plot_state_city
import qiskit.quantum_info as qi

from sklearn.utils import shuffle
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import QSVC
from sklearn.svm import SVC
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from QQQ import rbf_kernel, fidelity_kernel
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

    final_train_features, final_train_labels, final_test_features, final_test_labels = preprocess(opt)

    path = f"/home/sil/quantum/outputs/{opt.instance}-{opt.kernel}-{opt.feature_map}-{opt.numqubits}/{opt.reps}-{opt.trainnum}-{opt.testnum}-{opt.normalize}"

    if not os.path.exists(path):
        os.makedirs(path)
        print(f"'{path}' 폴더가 생성되었습니다.")
    else:
        print(f"'{path}' 폴더가 이미 존재합니다.")

    # train_features, train_labels = shuffle(train_features, train_labels, random_state=seed)
    # test_features, test_labels = shuffle(test_features, test_labels, random_state=seed)

    # train_features = train_features[0:opt.trainnum]
    # train_labels=train_labels[0:opt.trainnum]
    # test_features=test_features[0:opt.testnum]
    # test_labels=test_labels[0:opt.testnum]
    
    svc = SVC()
    svc.fit(final_train_features, final_train_labels)
    svc_score = svc.score(final_test_features, final_test_labels)

    print(f"Classical SVC classification test score: {svc_score}")

    if opt.instance != 'aer' and opt.instance != 'statevector':
        service = QiskitRuntimeService()
        backend = service.backend(opt.instance)
        if backend.simulator:
            options = Options(optimization_level=2, resilience_level=0)
        else:
            options = Options(optimization_level=3, resilience_level=1)
    else:
        service, backend, options = None, None, None

    if opt.feature_map == 'real_amp':
        feature_map_instance = RealAmplitudes(num_qubits=opt.numqubits, reps=opt.reps, entanglement='pairwise')
    elif opt.feature_map == 'zzfeature':
        feature_map_instance = ZZFeatureMap(feature_dimension=opt.numqubits, reps=opt.reps, entanglement='pairwise')
    else:
        raise ValueError(f"Unknown feature map: {opt.feature_map}")

    if opt.kernel == 'rbf_kernel':
        quantum_kernel = rbf_kernel(
            feature_map=feature_map_instance,
            quantum_instance=opt.instance,
            device=opt.device,
            save_path=path,
            service=service,
            backend=backend,
            options=options,
        )
    elif opt.kernel == 'fidelity_kernel':
        quantum_kernel = fidelity_kernel(
            feature_map=feature_map_instance,
            quantum_instance=opt.instance,
            device=opt.device,
            save_path=path,
            service=service,
            backend=backend,
            options=options,
        )
    elif opt.kernel == 'default_kernel':
        sampler = Sampler()
        fidelity = ComputeUncompute(sampler=sampler)
        quantum_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map_instance)
    else:
        raise ValueError(f"Unknown kernel: {opt.kernel}")
    
    qsvc = QSVC(quantum_kernel=quantum_kernel)
    if opt.kernel == 'default_kernel':
        t0 = time.time()
        print(f'Simulation queued: {time.strftime("%H:%M:%S")}')
    qsvc.fit(final_train_features, final_train_labels)
    if opt.kernel == 'default_kernel':
        t1 = time.time()
        print(f'Simulation done: {t1-t0:.2f}s')
        print(f'Simulation queued: {time.strftime("%H:%M:%S")}')
    qsvc_score = qsvc.score(final_test_features, final_test_labels)
    if opt.kernel == 'default_kernel':
        t2 = time.time()
        print(f'Simulation done: {t2-t1:.2f}s')

    print(f"Quantum SVC classification test score: {qsvc_score}")


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