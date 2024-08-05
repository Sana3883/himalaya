import torch
from sklearn.model_selection import train_test_split
from himalaya.backend import set_backend
from himalaya.ridge import solve_ridge_cv_svd
import os

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection:True'

# Set backend to torch_cuda
backend = set_backend("torch_cuda")

def my_func_new_1():
    
    Movie_Data = np.load('Data_Season1_shift4_sub1.npy', allow_pickle=True)
   
    fMRI_Data = np.load('fMRI_Data_v.npy', allow_pickle=True)

    print("****************Data_shape")
    print(Movie_Data.shape, fMRI_Data.shape)

  
    fMRI_Data = fMRI_Data.astype("float32")
    Movie_Data = Movie_Data.astype("float32")

    
    x_train, x_test, y_train, y_test = train_test_split(Movie_Data[0:10000, :], fMRI_Data[0:10000, :], test_size=0.1)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    x_train = backend.asarray(x_train)
    y_train = backend.asarray(y_train)

    alphas = [0.1, 1, 100, 200, 300, 400, 600, 800, 900, 1000, 1200]
    alphas = np.array(alphas)
    # Adjust batch sizes to avoid out of memory errors
    ridge_cv_results = solve_ridge_cv_svd(
        X=x_train,
        Y=y_train,
        alphas=alphas,
        cv=len(x_train),  # Set cv to the number of training samples (LOOCV)
        n_targets_batch=50,  # what is the best value? 
        n_alphas_batch=1      # ? TOdo
    )

    print("RidgeCV with solve_ridge_cv_svd done!")

    # Optionally clear GPU cache
    torch.cuda.empty_cache()

my_func_new_1()

