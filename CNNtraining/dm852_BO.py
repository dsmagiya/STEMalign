from autoscript_tem_microscope_client import TemMicroscopeClient
from autoscript_tem_microscope_client.enumerations import *
from autoscript_tem_microscope_client.structures import *
from Corrector_CEOS import CorrectorCommands
corrector = CorrectorCommands()
info = corrector.getInfo()

from matplotlib import pyplot as plot

import torch
import torch.nn as nn
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition import PosteriorMean
from botorch.acquisition import ExpectedImprovement
from botorch.models.transforms.outcome import Standardize
import gpytorch
import torchvision
from torch.nn import Module
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pickle
device = "cpu"



print("Starting dm852 Bayesian Optimization script...")

# Connect to AutoScript server
microscope = TemMicroscopeClient()
microscope.connect()
microscope.optics.defocus = 0
defocus_bound = 0.5*1e-7
order1_bound = 2*1e-7
order2_bound = 2*1e-6
# Change optical mode to STEM
microscope.optics.optical_mode = OpticalMode.STEM
# center the beam
microscope.optics.deflectors.image_tilt = Point(0,0)

# Read out microscope type and software version
system_info = microscope.service.system.name + ", version " + microscope.service.system.version
print("System info: " + system_info)

# Read out available detectors and cameras
camera_detectors = microscope.detectors.camera_detectors
print("Camera detectors: ", camera_detectors)
scanning_detectors = microscope.detectors.scanning_detectors
print("Scanning detectors: ", scanning_detectors)

print("System Info check finished successfully.")

# for camera_name in microscope.detectors.camera_detectors:
#     camera = microscope.detectors.get_camera_detector(camera_name)
#     print(camera.display_name)
#     print(camera_name)

# set CCD to capture Ronchigram
ceta = 'BM-Ceta'
image = microscope.acquisition.acquire_camera_image(ceta, 512, 1)
# plot.imshow(image.data, cmap="gray")
# plot.show()

model_name = 'DenseNet' #{VGG, ResNet, DenseNet}
if model_name == 'VGG':
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=True)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 1)
    model.load_state_dict(
        torch.load(r'C:\Users\Customer\PycharmProjects\dm852_BO_kraken\trained_models\trained_models\VGG16bn_25000_aperture0.pt', map_location=torch.device('cpu')))
    model.eval()
elif model_name == 'ResNet':
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model.load_state_dict(
        torch.load(r'C:\Users\Customer\PycharmProjects\dm852_BO_kraken\trained_models\trained_models\Resnet_25000_aperture0.pt', map_location=torch.device('cpu')))
    # if torch.cuda.is_available():
    #  model.to('cuda')
    model.eval()
elif model_name == 'DenseNet':
    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 1)
    model.load_state_dict(torch.load(r'C:\Users\Customer\PycharmProjects\dm852_BO_kraken\trained_models\trained_models\Densenet_25000_aperture0.pt', map_location=torch.device('cpu')))
    model.eval()

# if torch.cuda.is_available():
#     model.to('cuda')
# model.eval()

import cv2
from PIL import Image
def find_circle(img):
    # img is 2D
    #     img = 255*(img/np.max(img))
    #     img = np.dstack((img, img, img))
    #     img = img.astype(np.uint8)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    output = img.copy()
    circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1.3, 100, param1=100, param2 = 40, minRadius=40, maxRadius= 70)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            print(x, y, r)
    #plt.imshow(output)
    return circles

def scale_range (input, min, max):
    input += -(np.min(input))
    input /= np.max(input) / (max - min)
    input += min
    return input

from torchvision import transforms
transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

def image_formatting(image):
    frame = image.data.astype('float')
    frame = scale_range(frame, 0, 1)
    img_nparray = np.dstack((frame, frame, frame))
    # xyr = find_circle(np.uint8(img_nparray * 255))
    xyr = [[265,257,42]]
    img =  Image.fromarray(np.uint8(img_nparray * 255))
    img_cropped = torchvision.transforms.functional.crop(img, xyr[0][1] - (4 / 3) * xyr[0][2], xyr[0][0] - (4 / 3) * xyr[0][2], 2 * (4 / 3) * xyr[0][2], 2 * (4 / 3) * xyr[0][2])
    img_cropped = transform(img_cropped)

    return img_cropped.unsqueeze(0)

def acquire_and_pred(x):
    x = x.squeeze().tolist()
    microscope.optics.defocus = x[0] * 1e0
    #corrector.correctAberration(name='C1', value=[x[0], 0], select='coarse')
    corrector.correctAberration(name='A1', value=[x[1], x[2]], select='coarse')
    corrector.correctAberration(name='A2', value=[x[3], x[4]], select='coarse')
    corrector.correctAberration(name='B2', value=[x[5], x[6]], select='coarse')

    image = microscope.acquisition.acquire_camera_image(ceta, 512, 1.0)
    output_y = model(image_formatting(image).float().contiguous().to(device))
    y = (output_y).unsqueeze(-1).clone().detach()[0]

    microscope.optics.defocus = 0
    corrector.correctAberration(name='B2', value=[-x[5], -x[6]], select='coarse')
    corrector.correctAberration(name='A2', value=[-x[3], -x[4]], select='coarse')
    corrector.correctAberration(name='A1', value=[-x[1], -x[2]], select='coarse')
    #corrector.correctAberration(name='C1', value=[-x[0], 0], select='coarse')
    return image, 1-y

def generate_initial_data(n, model):
    train_ronch = []
    train_X = torch.tensor([])
    train_Y = torch.tensor([])
    ronch = []
    # generate training data
    for i in range(n):
        x = (torch.rand(1, 7, device=device))-0.5
        x = x*2
        x[0][0] = x[0][0] * defocus_bound
        x[0][[1,2]] = x[0][[1,2]] * order1_bound
        x[0][[3,4,5,6]] = x[0][[3,4,5,6]] * order2_bound

        image, y = acquire_and_pred(x)
        train_X = torch.cat([train_X, x])
        train_Y = torch.cat([train_Y, y])
        ronch.append(image.data)
    return train_X, train_Y, ronch

train_X, train_Y, ronch = generate_initial_data(n = 5, model = model)
savepath = r"C:\Users\Customer\PycharmProjects\dm852_BO_kraken\300kv_laceyCarbon\bo\\"
np.save(savepath + 'train' + '__ronch.npy', np.array(ronch))
np.save(savepath + 'train' + '__X.npy', np.array(train_X))
np.save(savepath + 'train' + '__Y.npy', np.array(train_Y))

## BO Settings
niter = 500
nrep = 1
option_standardize = True
kernel_name = 'Matern' #{RBF, Matern, DKL}
acq_name = 'EI' #{UCB, EI, PM}
ndim = 3
BO_filename_head =  '%dD_%diter_%dreps_%s_%s_%s_aperture0' % (ndim, niter, nrep, kernel_name, acq_name, model_name)

seen_X = torch.tensor([])
seen_Y = []
seen_rep = []
seen_ronch = []
best_seen_rep = []
best_par_rep = []
best_ronchigram_rep = []
best_seen_ronchigram = np.zeros([512, 512])


if option_standardize:
    outcome_transformer = Standardize( m = 1,
    batch_shape = torch.Size([]),
    min_stdv = 1e-08)

ronch_list = []
for irep in range(nrep):
    if (irep%5) == 0:
        microscope.stage.relative_move_safe(StagePosition(x=1e-7))

    print('% Run ' + str(irep + 1) + '/' + str(nrep))

    if kernel_name == 'Matern':
        if option_standardize:
            gp = SingleTaskGP(train_X, train_Y, outcome_transform=outcome_transformer)
        else:
            gp = SingleTaskGP(train_X, train_Y)
    elif kernel_name == 'RBF':
        if option_standardize:
            gp = SingleTaskGP(train_X, train_Y, outcome_transform=outcome_transformer,
                              covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()))
        else:
            gp = SingleTaskGP(train_X, train_Y,
                              covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()))
    else:
        print("------ Error ------ Not Compatible Kernel! - dm852")
        break

    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)
    bounds = torch.stack([-torch.ones(7, device=device)*0.999 , torch.ones(7, device=device)*0.999 ])
    bounds[:, 0] = bounds[:, 0] * defocus_bound
    bounds[:, [1,2]] = bounds[:, [1,2]] * order1_bound
    bounds[:, [3,4,5,6]] = bounds[:, [3,4,5,6]] * order2_bound
    best_value = np.array(train_Y[0][0])
    best_par = np.array(train_X[0])
    best_observed_value = [best_value]

    for i in range(1, 5):
        if np.array(train_Y[i]) > best_value:
            best_value = np.array(train_Y[i][0])
            best_par = np.array(train_X[i])
        best_observed_value.append(best_value)

    for iteration in range(niter):
        print('-- Run ' + str(irep + 1) + '/' + str(nrep) + ' ------------------  iteration ' + str(
            iteration) + '/' + str(niter))

        fit_gpytorch_model(mll)

        if acq_name == 'UCB':
            UCB = UpperConfidenceBound(gp, beta=2)
            candidate, acq_value = optimize_acqf(
                UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
            )
        elif acq_name == 'PM':
            PM = PosteriorMean(gp)
            candidate, acq_value = optimize_acqf(
                PM, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
            )
        elif acq_name == 'EI':
            EI = ExpectedImprovement(gp, best_f=best_value)
            candidate, acq_value = optimize_acqf(
                EI, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
            )
        new_x = candidate.detach()
        image, new_y = acquire_and_pred(new_x)
        ronch = image.data
        train_X = torch.cat([train_X, new_x])
        train_Y = torch.cat([train_Y, new_y])
        seen_ronch.append(ronch)

        if not best_observed_value:
            best_par = np.array(new_x[0])
            best_value = np.array(new_y[0])
            best_seen_ronchigram = result[0]
        elif new_y.item() > best_value:
            best_par = np.array(new_x[0])
            best_value = new_y.item()
            best_seen_ronchigram = image.data
        best_observed_value.append(best_value)

        # update GP model using dataset with new datapoint
        if option_standardize:
            gp = SingleTaskGP(train_X, train_Y, outcome_transform=outcome_transformer)
        else:
            gp = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

        print('best value: ', best_value)
        print('çurrent value: ', new_y.item())
        ronch_list.append(ronch)
        plt.imshow(image_formatting(image)[0,0,:], cmap="gray")
        plt.pause(0.05)
    plt.show()

    seen_rep.append(np.array(torch.transpose(train_Y[-niter:], 0, 1)))
    best_seen_rep.append(np.array(best_observed_value)[-niter:])
    best_par_rep.append(best_par)
    best_ronchigram_rep.append(best_seen_ronchigram)
    seen_Y = seen_rep
    seen_X = torch.cat((seen_X, train_X[-niter:]), 0)


np.save(savepath + BO_filename_head + '__best_seen_rep.npy', np.array(best_seen_rep))
np.save(savepath + BO_filename_head + '__X.npy', np.array(seen_X))
np.save(savepath + BO_filename_head + '__Y.npy', np.array(seen_Y))
np.save(savepath + BO_filename_head + '__ronch.npy', np.array(ronch_list))