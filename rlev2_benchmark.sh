echo PC
CUDA_VISIBLE_DEVICES=2 ./build/exe/src/rlev2/benchmark.cu.exe /data/brian/taxi-data/passenger_count_2byte.bin 2 >  ./result/RLEV2_V100_Passenget_count_2bresult.csv
echo PT
CUDA_VISIBLE_DEVICES=2 ./build/exe/src/rlev2/benchmark.cu.exe /data/brian/taxi-data/payment_type_2byte.bin 2 >  ./result/RLEV2_V100_Payment_type_2bresult.csv
echo COL0
CUDA_VISIBLE_DEVICES=2 ./build/exe/src/rlev2/benchmark.cu.exe /data/brian/mortgage/Performance_col0.bin 8 >  ./result/RLEV2_V100_Col0_result.csv
echo COL3
CUDA_VISIBLE_DEVICES=2 ./build/exe/src/rlev2/benchmark.cu.exe /data/brian/mortgage/Performance_col3.bin 4 >  ./result/RLEV2_V100_Col3_result.csv

echo A100 PC
CUDA_VISIBLE_DEVICES=1 ./build/exe/src/rlev2/benchmark.cu.exe /data/brian/taxi-data/passenger_count_2byte.bin 2 > ./result/RLEV2_A100_Passenget_count_2bresult.csv
echo A100 PT
CUDA_VISIBLE_DEVICES=1 ./build/exe/src/rlev2/benchmark.cu.exe /data/brian/taxi-data/payment_type_2byte.bin 2 >  ./result/RLEV2_A100_Payment_type_2bresult.csv

echo A100_COL0
CUDA_VISIBLE_DEVICES=1 ./build/exe/src/rlev2/benchmark.cu.exe /data/brian/mortgage/Performance_col0.bin 8 >  ./result/RLEV2_A100_Col0_result.csv
echo A100_COL3
CUDA_VISIBLE_DEVICES=1 ./build/exe/src/rlev2/benchmark.cu.exe /data/brian/mortgage/Performance_col3.bin 4 >  ./result/RLEV2_A100_Col3_result.csv



