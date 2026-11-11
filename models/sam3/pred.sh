# python predsam3.py --log_dir ./sam3/logs 2>&1 | tee ./log/sam2_$(date +%Y%m%d_%H%M%S).log

python predtransUNet.py --model_path ./sam3/work_dir/fundus_auto/TransUNet-Baseline-domain1,3_3000000/20260415-2051/transunet_best_model.pth --task_name fundus --val_domains 1,3 --domains 2,4 --method_name TransUNet_Fundus_domain_1
python predtransUNet.py --model_path ./sam3/work_dir/fundus_auto/TransUNet-Baseline-domain2,4_3000000/20260415-2114/transunet_best_model.pth --task_name fundus --val_domains 2,4 --domains 1,3 --method_name TransUNet_Fundus_domain_2
python predtransUNet.py --model_path ./sam3/work_dir/naso_auto/TransUNet-Baseline-domain1,4_3000000/20260415-2138/transunet_best_model.pth --task_name na --val_domains 1,4 --domains 2,3 --method_name TransUNet_NPC_domain_1
python predtransUNet.py --model_path ./sam3/work_dir/naso_auto/TransUNet-Baseline-domain2,3_3000000/20260415-2234/transunet_best_model.pth --task_name na --val_domains 2,3 --domains 1,4 --method_name TransUNet_NPC_domain_2
python predtransUNet.py --model_path ./sam3/work_dir/fundus_auto/TransUNet-Baseline-domain1,2_3000000/20260415-2348/transunet_best_model.pth --task_name poly --val_domains 1,2 --domains 3,4 --method_name TransUNet_PolypDB_domain_1
python predtransUNet.py --model_path ./sam3/work_dir/fundus_auto/TransUNet-Baseline-domain3,4_3000000/20260416-0006/transunet_best_model.pth --task_name poly --val_domains 3,4 --domains 1,2 --method_name TransUNet_PolypDB_domain_2
