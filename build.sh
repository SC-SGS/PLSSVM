mkdir Debug
cd Debug
cmake -DCMAKE_BUILD_TYPE=Debug ..
cd ..

mkdir Release
cd Release
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ..

