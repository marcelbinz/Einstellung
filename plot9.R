library(ggplot2)

pd <- position_dodge(.1)
cbPalette <- c("#009E73", "#F0E442")

d1<-read.csv('data/noadaptatione.csv')
d2<-read.csv('data/noadaptationd.csv')

dp1<-data.frame("X"=1:50, "Y1"=d1['X0'] * 100, "Y2"=d2['X0'] * 100)

ggp1 <- ggplot(dp1, aes(x=X)) +       # Create ggplot2 plot
  theme_minimal() +
  scale_color_manual(values=cbPalette)+
  labs(x = expression("Inverse temperature" ~ beta),
       y = "% response",
       color = "Type") +
  scale_y_continuous(limits = c(-5,105), expand = c(0, 0)) +
  scale_x_continuous(limits = c(1,50), expand = c(0, 0)) +
  geom_line(aes(y = X0.1, color = "D"), size=1.2) + 
  geom_line(aes(y = X0, color = "E"), size=1.2) +
  theme(text = element_text(size=12,  family="sans"),
        axis.text.x = element_text(size = 8))+
  theme(legend.position = "top") +
  ggtitle("Simulated effect (ablated model)")
ggp1            

ggsave(
  'figures/noadaption.pdf',
  width = 3.5,
  height = 3,
)

