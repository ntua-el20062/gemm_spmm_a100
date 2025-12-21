#for i in {1..6}; do
#  ./spmm_stream ../spmv/suitesparse_mats/ecology1/ecology1.mtx 32 100000 >> streamed_32.txt
#  ./spmm_csr ../spmv/suitesparse_mats/ecology1/ecology1.mtx 32 >> full_32.txt
#done

#for i in {1..6}; do
#  ./spmm_stream ../spmv/suitesparse_mats/ecology1/ecology1.mtx 64 100000 >> streamed_64.txt
#  ./spmm_csr ../spmv/suitesparse_mats/ecology1/ecology1.mtx 64  >> full_64.txt
#done

for i in {1..6}; do
  ./spmm_stream ../spmv/suitesparse_mats/ecology1/ecology1.mtx 128 100000 >> streamed_128.txt
  ./spmm_csr ../spmv/suitesparse_mats/ecology1/ecology1.mtx 128 >> full_128.txt
done

for i in {1..6}; do
  ./spmm_stream ../spmv/suitesparse_mats/ecology1/ecology1.mtx 256 100000 >> streamed_256.txt
  ./spmm_csr ../spmv/suitesparse_mats/ecology1/ecology1.mtx 256  >> full_256.txt
done

for i in {1..6}; do
  ./spmm_stream ../spmv/suitesparse_mats/ecology1/ecology1.mtx 512 100000 >> streamed_512.txt
  ./spmm_csr ../spmv/suitesparse_mats/ecology1/ecology1.mtx 512 >> full_512.txt
done

for i in {1..6}; do
  ./spmm_stream ../spmv/suitesparse_mats/ecology1/ecology1.mtx 1024 100000 >> streamed_1024.txt
  ./spmm_csr ../spmv/suitesparse_mats/ecology1/ecology1.mtx 1024 >> full_1024.txt
done

for i in {1..6}; do
  ./spmm_stream ../spmv/suitesparse_mats/ecology1/ecology1.mtx 2048 100000 >> streamed_2048.txt
  ./spmm_csr ../spmv/suitesparse_mats/ecology1/ecology1.mtx 2048 >> full_2048.txt
done

for i in {1..6}; do
  ./spmm_stream ../spmv/suitesparse_mats/ecology1/ecology1.mtx 4096 100000 >> streamed_4096.txt
  ./spmm_csr ../spmv/suitesparse_mats/ecology1/ecology1.mtx 4096 >> full_4096.txt
done

for i in {1..6}; do
  ./spmm_stream ../spmv/suitesparse_mats/ecology1/ecology1.mtx 8192 100000 >> streamed_8192.txt
  ./spmm_csr ../spmv/suitesparse_mats/ecology1/ecology1.mtx 8192 >> full_8192.txt
done

for i in {1..6}; do
  ./spmm_stream ../spmv/suitesparse_mats/ecology1/ecology1.mtx 12000 100000 >> streamed_12000.txt
  ./spmm_csr ../spmv/suitesparse_mats/ecology1/ecology1.mtx 12000 >> full_12000.txt
done


for i in {1..6}; do
  ./spmm_stream ../spmv/suitesparse_mats/ecology1/ecology1.mtx 25000 100000 >> streamed_25000.txt
  ./spmm_csr ../spmv/suitesparse_mats/ecology1/ecology1.mtx 25000 >> full_25000.txt
done

for i in {1..6}; do
  ./spmm_stream ../spmv/suitesparse_mats/ecology1/ecology1.mtx 35000 100000 >> streamed_35000.txt
  ./spmm_csr ../spmv/suitesparse_mats/ecology1/ecology1.mtx 35000 >> full_35000.txt
done

for i in {1..6}; do
  ./spmm_stream ../spmv/suitesparse_mats/ecology1/ecology1.mtx 40000 100000 >> streamed_40000.txt
  ./spmm_csr ../spmv/suitesparse_mats/ecology1/ecology1.mtx 40000 >> full_40000.txt
done

for i in {1..6}; do
  ./spmm_stream ../spmv/suitesparse_mats/ecology1/ecology1.mtx 45000 100000 >> streamed_45000.txt
  ./spmm_csr ../spmv/suitesparse_mats/ecology1/ecology1.mtx 45000 >> full_45000.txt
done
