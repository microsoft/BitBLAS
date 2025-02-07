cd Figure10-A6000-end2end; python plot_figures.py --reproduce; cd ..
cd Figure11-A100-MemoryUsage; python plot_figures.py --reproduce; cd ..
cd Figure12-A100-Operator; python plot_figures.py --reproduce; cd ..
cd Figure13-A100-OpitimizationBreakDown; python plot_figures.py; cd ..
cd Figure14-A100-ScalingBitWidth; python plot_figures.py --reproduce; cd ..
cd Figure15-MI250-end2end; python plot_figures.py --reproduce; cd ..
cd Figure8-A100-end2end; python plot_figures.py --reproduce; cd ..
cd Figure9-V100-end2end; python plot_figures.py --reproduce; cd ..
cd PPL-Latency; python plot_figures.py; cd ..
cd 4090-Operator; python plot_figures.py; cd ..

rm -r reproduce
mkdir reproduce
cp -r Figure8-A100-end2end/pdf/* reproduce/
cp -r Figure9-V100-end2end/pdf/* reproduce/
cp -r Figure10-A6000-end2end/pdf/* reproduce/
cp -r Figure11-A100-MemoryUsage/pdf/* reproduce/
cp -r Figure12-A100-Operator/pdf/* reproduce/
cp -r Figure13-A100-OpitimizationBreakDown/pdf/* reproduce/
cp -r Figure14-A100-ScalingBitWidth/pdf/* reproduce/
cp -r Figure15-MI250-end2end/pdf/* reproduce/
cp -r PPL-Latency/pdf/* reproduce/
cp -r 4090-Operator/pdf/* reproduce/
