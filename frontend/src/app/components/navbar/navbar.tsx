// components/Navbar.js
import React from 'react';
import { AppBar, Toolbar, IconButton, Typography, Drawer, List, ListItem, ListItemText, Box } from '@mui/material';
import Link from 'next/link';
import Image from 'next/image';
import Logo from '../../../../public/images/logo.svg';
import './navbar.css'


const Navbar = async () => {
  return (
    <AppBar position="static">
      <Toolbar>
          <Box sx={{ display: 'flex', flexGrow: 1, gap: 10, alignItems: 'center', height: 60}}>
          <Link href="/" passHref>
              <div style={{display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10}}>
              <Image className='navbar-logo' src={Logo} alt={'profile'} width='40' height='40'></Image>
              <Typography variant="h6" sx={{font: 'Gurajada'}}>

               Cinemap
               </Typography>
                
              </div>
              </Link>
            <Typography variant="h6">
              <Link href="/about" passHref>
              Contact
              </Link>
            </Typography>
              <Typography variant="h6" sx={{ marginLeft: 'auto', justifySelf: 'flex-end', backgroundColor: 'green' }}>
              <Link href="/api/auth/signout" passHref>
                Log Out
              </Link>
            </Typography>
          </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;
