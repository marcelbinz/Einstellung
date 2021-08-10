import torch
import numpy as np
import pandas as pd

# files for full model
num_correct, num_d_solution, num_e_solution, klds = torch.load('data/exp0_gamma_0.05_bounded_True.pth')
num_correct_control, num_d_solution_control, num_e_solution_control, klds_control = torch.load('data/exp1_gamma_0.05_bounded_True.pth')

avg_e1 = pd.DataFrame(num_e_solution[:, :, :, [5, 6]].mean(-1).mean(0).t().numpy())
avg_e2 = pd.DataFrame(num_e_solution[:, :, :, [8, 9]].mean(-1).mean(0).t().numpy())

avg_e1.to_csv('data/esolution1.csv')
avg_e2.to_csv('data/esolution2.csv')

avg_e1_control = pd.DataFrame(num_e_solution_control[:, :, :, [0, 1]].mean(-1).mean(0).t().numpy())
avg_e2_control = pd.DataFrame(num_e_solution_control[:, :, :, [3, 4]].mean(-1).mean(0).t().numpy())

avg_e1_control.to_csv('data/esolution_control1.csv')
avg_e2_control.to_csv('data/esolution_control2.csv')

avg_kld = pd.DataFrame(klds[:, :, :, [5, 6, 8, 9]].mean(-1).mean(0).t().numpy())
avg_kld.to_csv('data/klds.csv')

avg_correct = pd.DataFrame(num_correct[:, :, :, [5, 6, 8, 9]].mean(-1).mean(0).t().numpy())
avg_correct.to_csv('data/correct.csv')

avg_d = pd.DataFrame(num_d_solution[:, :, :, [5, 6, 8, 9]].mean(-1).mean(0).t().numpy())
avg_d.to_csv('data/dsolutions.csv')

avg_e =  pd.DataFrame(num_e_solution[:, :, :, [5, 6, 8, 9]].mean(-1).mean(0).t().numpy())
avg_e.to_csv('data/esolutions.csv')

avg_d = pd.DataFrame(num_d_solution_control[:, :, :, [0, 1, 3, 4]].mean(-1).mean(0).t().numpy())
avg_d.to_csv('data/dsolutionscontrol.csv')

avg_e = pd.DataFrame(num_e_solution_control[:, :, :, [0, 1, 3, 4]].mean(-1).mean(0).t().numpy())
avg_e.to_csv('data/esolutionscontrol.csv')

num_correct_less_e, num_d_solution_less_e, num_e_solution_less_e, klds_less_e = torch.load('data/exp2_gamma_0.05_bounded_True.pth')

avg_e1_less_e = pd.DataFrame(num_e_solution_less_e[:, :, :, [2, 3]].mean(-1).mean(0).t().numpy())
avg_e2_less_e = pd.DataFrame(num_e_solution_less_e[:, :, :, [5, 6]].mean(-1).mean(0).t().numpy())

avg_e1_less_e.to_csv('data/e1short.csv')
avg_e2_less_e.to_csv('data/e2short.csv')

num_correct_more_e, num_d_solution_more_e, num_e_solution_more_e, klds_more_e = torch.load('data/exp3_gamma_0.05_bounded_True.pth')
avg_e1_more_e = pd.DataFrame(num_e_solution_more_e[:, :, :, [10, 11]].mean(-1).mean(0).t().numpy())
avg_e2_more_e = pd.DataFrame(num_e_solution_more_e[:, :, :, [13, 14]].mean(-1).mean(0).t().numpy())

avg_e1_more_e.to_csv('data/e1long.csv')
avg_e2_more_e.to_csv('data/e2long.csv')

print(avg_e1_more_e.shape)

num_correct_alternating, num_d_solution_alternating, num_e_solution_alternating, klds_alternating = torch.load('data/exp4_gamma_0.05_bounded_True.pth')

avg_e1_alternating = pd.DataFrame(num_e_solution_alternating[:, :, :, [6, 7]].mean(-1).mean(0).t().numpy())
avg_e1_alternating.to_csv('data/ealternating.csv')

avg_e1 = pd.DataFrame(num_e_solution[:, :, :, [5, 6]].mean(-1).mean(0).t().numpy()[[25], :])
avg_d1 = pd.DataFrame(num_d_solution[:, :, :, [5, 6]].mean(-1).mean(0).t().numpy()[[25], :])
avg_e1.to_csv('data/dev_e1.csv')
avg_d1.to_csv('data/dev_d1.csv')


# files for rational model
num_correct, num_d_solution, num_e_solution, klds = torch.load('data/exp0_gamma_0.05_bounded_False.pth')
num_correct_control, num_d_solution_control, num_e_solution_control, klds_control = torch.load('data/exp1_gamma_0.05_bounded_False.pth')

avg_e1 = pd.DataFrame(num_e_solution[:, :, :, [5, 6]].mean(-1).mean(0).t().numpy())
avg_e2 = pd.DataFrame(num_e_solution[:, :, :, [8, 9]].mean(-1).mean(0).t().numpy())

avg_e1.to_csv('data/esolution1_no_mental.csv')
avg_e2.to_csv('data/esolution2_no_mental.csv')

avg_e1_control = pd.DataFrame(num_e_solution_control[:, :, :, [0, 1]].mean(-1).mean(0).t().numpy())
avg_e2_control = pd.DataFrame(num_e_solution_control[:, :, :, [3, 4]].mean(-1).mean(0).t().numpy())

avg_e1_control.to_csv('data/esolution_control1_no_mental.csv')
avg_e2_control.to_csv('data/esolution_control2_no_mental.csv')

avg_kld = pd.DataFrame(klds[:, :, :, [5, 6, 8, 9]].mean(-1).mean(0).t().numpy())
avg_kld.to_csv('data/klds_no_mental.csv')

avg_correct = pd.DataFrame(num_correct[:, :, :, [5, 6, 8, 9]].mean(-1).mean(0).t().numpy())
avg_correct.to_csv('data/correct_no_mental.csv')

avg_d = pd.DataFrame(num_d_solution[:, :, :, [5, 6, 8, 9]].mean(-1).mean(0).t().numpy())
avg_d.to_csv('data/dsolutions_no_mental.csv')

avg_e =  pd.DataFrame(num_e_solution[:, :, :, [5, 6, 8, 9]].mean(-1).mean(0).t().numpy())
avg_e.to_csv('data/esolutions_no_mental.csv')

avg_d = pd.DataFrame(num_d_solution_control[:, :, :, [0, 1, 3, 4]].mean(-1).mean(0).t().numpy())
avg_d.to_csv('data/dsolutionscontrol_no_mental.csv')

avg_e = pd.DataFrame(num_e_solution_control[:, :, :, [0, 1, 3, 4]].mean(-1).mean(0).t().numpy())
avg_e.to_csv('data/esolutionscontrol_no_mental.csv')

num_correct_less_e, num_d_solution_less_e, num_e_solution_less_e, klds_less_e = torch.load('data/exp2_gamma_0.05_bounded_False.pth')

avg_e1_less_e = pd.DataFrame(num_e_solution_less_e[:, :, :, [2, 3]].mean(-1).mean(0).t().numpy())
avg_e2_less_e = pd.DataFrame(num_e_solution_less_e[:, :, :, [5, 6]].mean(-1).mean(0).t().numpy())

avg_e1_less_e.to_csv('data/e1short_no_mental.csv')
avg_e2_less_e.to_csv('data/e2short_no_mental.csv')

num_correct_more_e, num_d_solution_more_e, num_e_solution_more_e, klds_more_e = torch.load('data/exp3_gamma_0.05_bounded_False.pth')
avg_e1_more_e = pd.DataFrame(num_e_solution_more_e[:, :, :, [10, 11]].mean(-1).mean(0).t().numpy())
avg_e2_more_e = pd.DataFrame(num_e_solution_more_e[:, :, :, [13, 14]].mean(-1).mean(0).t().numpy())

avg_e1_more_e.to_csv('data/e1long_no_mental.csv')
avg_e2_more_e.to_csv('data/e2long_no_mental.csv')

num_correct_alternating, num_d_solution_alternating, num_e_solution_alternating, klds_alternating = torch.load('data/exp4_gamma_0.05_bounded_False.pth')

avg_e1_alternating = pd.DataFrame(num_e_solution_alternating[:, :, :, [6, 7]].mean(-1).mean(0).t().numpy())
avg_e1_alternating.to_csv('data/ealternating_no_mental.csv')

avg_e1 = pd.DataFrame(num_e_solution[:, :, :, [5, 6]].mean(-1).mean(0).t().numpy()[[0], :])
avg_d1 = pd.DataFrame(num_d_solution[:, :, :, [5, 6]].mean(-1).mean(0).t().numpy()[[0], :])
avg_e1.to_csv('data/dev_e1_no_mental.csv')
avg_d1.to_csv('data/dev_d1_no_mental.csv')

# files for no adaptation
num_correct, num_d_solution, num_e_solution, klds = torch.load('data/exp0_gamma_0.05_bounded_True.pth')
num_e_solution_full_total = num_e_solution[:, :, :, [5, 6, 8, 9]].mean(-1).mean(0).t()
num_d_solution_full_total = num_d_solution[:, :, :, [5, 6, 8, 9]].mean(-1).mean(0).t()

x_np = num_e_solution_full_total[:, 0].numpy()
x_df = pd.DataFrame(x_np)
x_df.to_csv('data/noadaptatione.csv')

x_np = num_d_solution_full_total[:, 0].numpy()
x_df = pd.DataFrame(x_np)
x_df.to_csv('data/noadaptationd.csv')

# files for no phyisical effort model
num_correct, num_d_solution, num_e_solution, klds = torch.load('data/exp0_gamma_0.0_bounded_True.pth')
num_correct_control, num_d_solution_control, num_e_solution_control, klds_control = torch.load('data/exp1_gamma_0.0_bounded_True.pth')

avg_e1 = pd.DataFrame(num_e_solution[:, :, :, [5, 6]].mean(-1).mean(0).t().numpy())
avg_e2 = pd.DataFrame(num_e_solution[:, :, :, [8, 9]].mean(-1).mean(0).t().numpy())

avg_e1.to_csv('data/esolution1_no_physical.csv')
avg_e2.to_csv('data/esolution2_no_physical.csv')

avg_e1_control = pd.DataFrame(num_e_solution_control[:, :, :, [0, 1]].mean(-1).mean(0).t().numpy())
avg_e2_control = pd.DataFrame(num_e_solution_control[:, :, :, [3, 4]].mean(-1).mean(0).t().numpy())

avg_e1_control.to_csv('data/esolution_control1_no_physical.csv')
avg_e2_control.to_csv('data/esolution_control2_no_physical.csv')

avg_kld = pd.DataFrame(klds[:, :, :, [5, 6, 8, 9]].mean(-1).mean(0).t().numpy())
avg_kld.to_csv('data/klds_no_physical.csv')

avg_correct = pd.DataFrame(num_correct[:, :, :, [5, 6, 8, 9]].mean(-1).mean(0).t().numpy())
avg_correct.to_csv('data/correct_no_physical.csv')

avg_d = pd.DataFrame(num_d_solution[:, :, :, [5, 6, 8, 9]].mean(-1).mean(0).t().numpy())
avg_d.to_csv('data/dsolutions_no_physical.csv')

avg_e =  pd.DataFrame(num_e_solution[:, :, :, [5, 6, 8, 9]].mean(-1).mean(0).t().numpy())
avg_e.to_csv('data/esolutions_no_physical.csv')

avg_d = pd.DataFrame(num_d_solution_control[:, :, :, [0, 1, 3, 4]].mean(-1).mean(0).t().numpy())
avg_d.to_csv('data/dsolutionscontrol_no_physical.csv')

avg_e = pd.DataFrame(num_e_solution_control[:, :, :, [0, 1, 3, 4]].mean(-1).mean(0).t().numpy())
avg_e.to_csv('data/esolutionscontrol_no_physical.csv')

num_correct_less_e, num_d_solution_less_e, num_e_solution_less_e, klds_less_e = torch.load('data/exp2_gamma_0.0_bounded_True.pth')

avg_e1_less_e = pd.DataFrame(num_e_solution_less_e[:, :, :, [2, 3]].mean(-1).mean(0).t().numpy())
avg_e2_less_e = pd.DataFrame(num_e_solution_less_e[:, :, :, [5, 6]].mean(-1).mean(0).t().numpy())

avg_e1_less_e.to_csv('data/e1short_no_physical.csv')
avg_e2_less_e.to_csv('data/e2short_no_physical.csv')

num_correct_more_e, num_d_solution_more_e, num_e_solution_more_e, klds_more_e = torch.load('data/exp3_gamma_0.0_bounded_True.pth')
avg_e1_more_e = pd.DataFrame(num_e_solution_more_e[:, :, :, [10, 11]].mean(-1).mean(0).t().numpy())
avg_e2_more_e = pd.DataFrame(num_e_solution_more_e[:, :, :, [13, 14]].mean(-1).mean(0).t().numpy())

avg_e1_more_e.to_csv('data/e1long_no_physical.csv')
avg_e2_more_e.to_csv('data/e2long_no_physical.csv')

num_correct_alternating, num_d_solution_alternating, num_e_solution_alternating, klds_alternating = torch.load('data/exp4_gamma_0.0_bounded_True.pth')

avg_e1_alternating = pd.DataFrame(num_e_solution_alternating[:, :, :, [6, 7]].mean(-1).mean(0).t().numpy())
avg_e1_alternating.to_csv('data/ealternating_no_physical.csv')

avg_e1 = pd.DataFrame(num_e_solution[:, :, :, [5, 6]].mean(-1).mean(0).t().numpy()[[25], :])
avg_d1 = pd.DataFrame(num_d_solution[:, :, :, [5, 6]].mean(-1).mean(0).t().numpy()[[25], :])
avg_e1.to_csv('data/dev_e1_no_physical.csv')
avg_d1.to_csv('data/dev_d1_no_physical.csv')
