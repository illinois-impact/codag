echo COL0
CUDA_VISIBLE_DEVICES=2 ./build/exe/src/rlev1/benchmark.cu.exe -f /data/brian/mortgage/Performance_col0.bin -t 8 >  ./result/RLEV1_V100_Col0_result.csv
echo COL3
CUDA_VISIBLE_DEVICES=2 ./build/exe/src/rlev1/benchmark.cu.exe -f /data/brian/mortgage/Performance_col3.bin -t 4 >  ./result/RLEV1_V100_Col3_result.csv
echo PC
CUDA_VISIBLE_DEVICES=2 ./build/exe/src/rlev1/benchmark.cu.exe -f  /data/brian/taxi-data/passenger_count_2byte.bin -t 2 > ./result/RLEV1_V100_Passenger_count_2bresult.csv
echo PT
CUDA_VISIBLE_DEVICES=2 ./build/exe/src/rlev1/benchmark.cu.exe -f  /data/brian/taxi-data/payment_type_2byte.bin -t 2 >   ./result/RLEV1_V100_Payment_type_2bresult.csv

echo A100
echo COL0
CUDA_VISIBLE_DEVICES=1 ./build/exe/src/rlev1/benchmark.cu.exe -f /data/brian/mortgage/Performance_col0.bin -t 8 >  ./result/RLEV1_A100_Col0_result.csv
echo COL3
CUDA_VISIBLE_DEVICES=1 ./build/exe/src/rlev1/benchmark.cu.exe -f /data/brian/mortgage/Performance_col3.bin -t 4 >  ./result/RLEV1_A100_Col3_result.csv
echo PC
CUDA_VISIBLE_DEVICES=1 ./build/exe/src/rlev1/benchmark.cu.exe -f  /data/brian/taxi-data/passenger_count_2byte.bin -t 2 > ./result/RLEV1_A100_Passenger_count_2bresult.csv
echo PT
CUDA_VISIBLE_DEVICES=1 ./build/exe/src/rlev1/benchmark.cu.exe -f  /data/brian/taxi-data/payment_type_2byte.bin -t 2 >   ./result/RLEV1_A100_Payment_type_2bresult.csv

