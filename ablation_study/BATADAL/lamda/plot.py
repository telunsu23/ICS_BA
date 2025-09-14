import matplotlib.pyplot as plt

lamda = [0.5, 1, 1.5, 2, 2.5]
br_values = [65.2 , 50.5, 59.1, 58.6, 13.8]
asr_values = [76.37 , 86.13, 89.58, 95.9, 100]

plt.figure(figsize=(8, 5))

plt.plot(lamda,
         br_values,
         'b-o',
         markersize=5,
         label='BR')

plt.plot(lamda,
         asr_values,
         'r^-',
         markersize=5,
         label='ASR')

plt.xlabel(r'$\lambda_1$ / $\lambda_2$', fontsize=16)
plt.ylabel('BR/ASR (%)',fontsize=16)

plt.xlim(0.4, 2.6)
plt.ylim(-2.5, 107.5)

plt.xticks(lamda,fontsize=12)
plt.yticks(range(0, 101, 20),fontsize=12)

plt.grid(axis='y',
        linestyle='--',
        alpha=0.7)

plt.legend(loc='upper left',fontsize=10)

plt.tight_layout()

output_path = 'D:\OneDrive\Desktop\Dynamic-Stealthy-Backdoor-Attack-ICS\lamda.pdf'
plt.savefig(
    output_path,
    format='pdf',
    dpi=1200,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none',
)

plt.show()