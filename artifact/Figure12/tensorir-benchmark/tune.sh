export CUDA_VISIBLE_DEVICES=2
export TVM_HOME=/root/unity
export PYTHONPATH=$TVM_HOME/python
# python3 meta.py --M 1 --N 16384 --K 16384 --trails 20
# python3 meta.py --M 1 --N 43008 --K 14336 --trails 20
# python3 meta.py --M 1 --N 14336 --K 14336 --trails 20
# python3 meta.py --M 1 --N 57344 --K 14336 --trails 20
# python3 meta.py --M 1 --N 14336 --K 57344 --trails 20
# python3 meta.py --M 1 --N 9216 --K 9216 --trails 20
# python3 meta.py --M 1 --N 36864 --K 9216 --trails 20
# python3 meta.py --M 1 --N 9216 --K 36864 --trails 20
# python3 meta.py --M 1 --N 8192 --K 8192 --trails 20
# python3 meta.py --M 1 --N 22016 --K 8192 --trails 20
# python3 meta.py --M 1 --N 8192 --K 22016 --trails 20

# python3 meta_nn.py --M 16384 --N 16384 --K 16384 --trails 1000
# python3 meta_nn.py --M 8192 --N 43008 --K 14336 --trails 1000
# python3 meta_nn.py --M 8192 --N 14336 --K 14336 --trails 1000
# python3 meta_nn.py --M 8192 --N 57344 --K 14336 --trails 1000
# python3 meta_nn.py --M 8192 --N 14336 --K 57344 --trails 1000
# python3 meta_nn.py --M 8192 --N 9216 --K 9216 --trails 1000
# python3 meta_nn.py --M 8192 --N 36864 --K 9216 --trails 1000
# python3 meta_nn.py --M 8192 --N 9216 --K 36864 --trails 1000
# python3 meta_nn.py --M 8192 --N 22016 --K 8192 --trails 1000
# python3 meta_nn.py --M 8192 --N 8192 --K 22016 --trails 1000
# python3 meta_nn.py --M 8192 --N 8192 --K 8192 --trails 1000
# python3 meta_nn.py --M 8192 --N 28672 --K 8192 --trails 1000
# python3 meta_nn.py --M 8192 --N 8192 --K 22016 --trails 1000


# python3 meta_nt.py --M 16384 --N 16384 --K 16384 --trails 1000
# python3 meta_nt.py --M 8192 --N 43008 --K 14336 --trails 1000
# python3 meta_nt.py --M 8192 --N 14336 --K 14336 --trails 1000
# python3 meta_nt.py --M 8192 --N 57344 --K 14336 --trails 1000
# python3 meta_nt.py --M 8192 --N 14336 --K 57344 --trails 1000
# python3 meta_nt.py --M 8192 --N 9216 --K 9216 --trails 1000
# python3 meta_nt.py --M 8192 --N 36864 --K 9216 --trails 1000
# python3 meta_nt.py --M 8192 --N 9216 --K 36864 --trails 1000
# python3 meta_nt.py --M 8192 --N 22016 --K 8192 --trails 1000
# python3 meta_nt.py --M 8192 --N 8192 --K 22016 --trails 1000
# python3 meta_nt.py --M 8192 --N 8192 --K 8192 --trails 1000
# python3 meta_nt.py --M 8192 --N 28672 --K 8192 --trails 1000
# python3 meta_nt.py --M 8192 --N 8192 --K 22016 --trails 1000




python3 meta_nt_int8.py --M 16384 --N 16384 --K 16384 --trails 1000
python3 meta_nt_int8.py --M 8192 --N 43008 --K 14336 --trails 1000
python3 meta_nt_int8.py --M 8192 --N 14336 --K 14336 --trails 1000
python3 meta_nt_int8.py --M 8192 --N 57344 --K 14336 --trails 1000
python3 meta_nt_int8.py --M 8192 --N 14336 --K 57344 --trails 1000
python3 meta_nt_int8.py --M 8192 --N 9216 --K 9216 --trails 1000
python3 meta_nt_int8.py --M 8192 --N 36864 --K 9216 --trails 1000
python3 meta_nt_int8.py --M 8192 --N 9216 --K 36864 --trails 1000
python3 meta_nt_int8.py --M 8192 --N 22016 --K 8192 --trails 1000
python3 meta_nt_int8.py --M 8192 --N 8192 --K 22016 --trails 1000
python3 meta_nt_int8.py --M 8192 --N 8192 --K 8192 --trails 1000
python3 meta_nt_int8.py --M 8192 --N 28672 --K 8192 --trails 1000
python3 meta_nt_int8.py --M 8192 --N 8192 --K 22016 --trails 1000
