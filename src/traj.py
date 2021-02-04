import numpy as np

def triTraj(wps, data, traj_type='poly'):
    '''
    inputs : 
    -> wps, numpy array x|y

    return :
    -> wps_traj, traj_def
    '''

    coords, l, org, scale = data['coords'], data['l'], data['org'], data['scale']


    a, b = wps[-2, :], wps[-1, :]
    angle = np.arctan2(b[1]-a[1], b[0]-a[0])
    c = np.array(((np.cos(angle), np.sin(angle))))
    wps = np.vstack((wps, c))

    wps_traj = np.array([wps[0, :]])
    traj_def = []
    traj_file_scaled = []

    if traj_type == 'poly':
        k = 2
    else:
        k = 3 


    for i in range(wps.shape[0]-k):

        # angles des deux segments
        phi1 = np.arctan2(wps[i+1, 1] - wps[i, 1], wps[i+1, 0] - wps[i, 0])
        phi2 = np.arctan2(wps[i+2, 1] - wps[i+1, 1], wps[i+2, 0] - wps[i+1, 0])
        
        # calcul point pour ligne
        wps_traj = np.vstack((wps_traj, wps[i+1, :] + np.array([l*np.cos(phi1), l*np.sin(phi1)])))
        traj_def.append(('line', wps_traj[-2, 0], wps_traj[-2, 1], wps_traj[-1, 0], wps_traj[-1, 1]))
        traj_file_scaled.append(('line', (wps_traj[-2, 0]-org[0])*scale, -(wps_traj[-2, 1]-org[1])*scale, (wps_traj[-1, 0]-org[0])*scale, -(wps_traj[-1, 1]-org[1])*scale))

        # calcul arc de cercle
        wps_traj = np.vstack((wps_traj, wps[i+1, :] + np.array([l*np.cos(phi2), l*np.sin(phi2)])))
        a1, a2 = -phi1*180/np.pi, (phi1-phi2)*180/np.pi
        if np.abs(a2) > 180:
            a2 = -np.sign(a2)*(360 - np.abs(a2))
        traj_def.append(('arc', wps[i+1, 0], wps[i+1, 1], a1, a2, l))
        traj_file_scaled.append(('arc', (wps[i+1, 0]-org[0])*scale, -(wps[i+1, 1]-org[1])*scale, phi1+np.pi/2, phi2+np.pi/2, l*scale))

 

    fichier = open("data_traj.txt", "w")
    fichier.write('coords ')
    fichier.write(coords)
    fichier.write('\n')
    for el in traj_file_scaled:
        fichier.write(' '.join(str(e) for e in el))
        fichier.write('\n')
    fichier.close()

    return wps_traj, traj_def

