import numpy as np

def runXNMF(V,nRuns,W01,H01,H02,W2,r):  
    mu,sigma=0,0.1
    m,n=V.shape[0],V.shape[1]
    errors=np.zeros((nRuns+1,2))
    r2=W2.shape[1]
    if not W01.size:
        W1 = np.random.normal(mu,sigma,size=(m,r));W1=np.absolute(W1)
        W1=W1*np.sqrt((np.mean(V)/r))/(np.mean(W1))
    else:
        W1=W01
    if not H01.size:
        H1 = np.random.normal(mu,sigma,size=(r,n));H1=np.absolute(H1)
        H1=H1*np.sqrt((np.mean(V)/r))/(np.mean(H1))
    else:
        H1=H01
    if not H02.size:
        H2 = np.random.normal(mu,sigma,size=(r2,n));H2=np.absolute(H2)
        H2=H2*np.sqrt((np.mean(V)/r))/(np.mean(H2))
    else:
        H2=H02
    W1store=[W1]
    H1store=[H1]
    H2store=[H2]
    

    V_pred=np.matmul(W1,H1)+np.matmul(W2,H2)
    errors[0,0]=0
    errors[0,1]=np.sum((V-V_pred)*(V-V_pred))
    for i in range(nRuns):
    # Update for the Hs
        W1,H1,H2=xMU1(W1,H1,W2,H2,V,m,n,r)
        V_pred=np.matmul(W1,H1)+np.matmul(W2,H2)
        errors[i+1,0]=i+1
        errors[i+1,1]=np.sum((V-V_pred)*(V-V_pred))
        print(i,errors[i,1]) 
    W1store.append(W1)
    H1store.append(H1)
    H2store.append(H2)
     
    return errors,W1,H1,H2,W2

def xMU1(W1,H1,W2,H2,V,m,n,r):
    H1 = H1*((np.matmul(np.transpose(W1),V))/(np.matmul(np.matmul(np.transpose(W1),W1),H1)+np.matmul(np.matmul(np.transpose(W1),W2),H2)+1e-9))
    numerator=(np.matmul(V,np.transpose(H1)))
    denominator=np.matmul(np.matmul(W1,H1),np.transpose(H1))+np.matmul(np.matmul(W2,H2),np.transpose(H1))+1e-9
    W1=W1*(numerator/denominator)
    numerator2=np.matmul(np.transpose(W2),V)
    denominator2=np.matmul(np.matmul(np.transpose(W2),W2),H2)+np.matmul(np.matmul(np.transpose(W2),W1),H1)+1e-9
    H2=H2*(numerator2/denominator2)
    return W1,H1,H2









