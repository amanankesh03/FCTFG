# syncnet512 = SyncNet512().to(device).eval()

# def get_sync_loss512(mel, g):
    
#     g_current = g[:, :, :, g.size(3)//2:]  #torch.Size([18, 3, 5, 96, 192])    # Get lower part of face,           
#     #print(g_current.size())
#     g_current = torch.cat([g_current[:, :, i] for i in range(syncnet_T)], dim=1) #torch.Size([18, 15, 48, 96])  # B, 3 * T, H//2, W    
#     #print(g_current.size())
#     a, v = syncnet512(mel, g_current)
#     y = torch.ones(g_current.size(0), 1).float().to(device)
#     #print(cosine_loss(a, v, y))
#     #print("End syncnet loss")
#     return cosine_loss(a, v, y)



# logloss = nn.BCELoss()

# def cosine_loss(a, v, y):
#     d = nn.functional.cosine_similarity(a, v)	# returns 1 if the feature vectors are similar
#     loss = logloss(d.unsqueeze(1), y)           # y has been assigned as all 1s so, logloss is 0 only if d and y are same, that is a and v are similar
#     return loss