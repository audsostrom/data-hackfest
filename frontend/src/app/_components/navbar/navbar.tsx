'use client';

// components/Navbar.js
import React, {useEffect, useState} from 'react';
import { AppBar, Toolbar, IconButton, Typography, Drawer, List, ListItem, ListItemText, Box } from '@mui/material';
import Link from 'next/link';
import Image from 'next/image';
import Logo from '../../../../public/images/logo.svg';
import Home from '../../../../public/images/home.svg';
import Friends from '../../../../public/images/people.svg';
import Bucket from '../../../../public/images/bucket.svg';
import SearchBar from "../search-bar";
import './navbar.css'
import {useSession} from "next-auth/react";


const Navbar = () => {
    const { data: session, status } = useSession();

    return (
    <AppBar position="static">
      <Toolbar>
          <Box sx={{ display: 'flex', flexGrow: 1, gap: 3, alignItems: 'center', height: 60}}>
          <Link href="/" passHref>
              <div style={{display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10}}>
              <Image className='navbar-logo' src={Logo} alt={'profile'} width='40' height='40'></Image>
              <Typography variant="h6" sx={{font: 'Gurajada'}}>

               Cinemap
               </Typography>
                
              </div>
              </Link>
            <Link style={{ marginLeft: 'auto', justifySelf: 'flex-end'}} href="/home" passHref>
              <Image className='home-logo' style={{ marginLeft: 'auto', justifySelf: 'flex-end'}} src={Home} alt={'profile'} width='30' height='30'></Image>
              </Link>

            <SearchBar />
            <Link href="/bucketlist" passHref>
              <Image className='navbar-logo' src={Bucket} alt={'profile'} width='30' height='30'></Image>
              </Link>
              <Link href={session && session.user ? "/account/" + session.user.id : '/login'} passHref>
              <Image className='navbar-logo' src={Friends} alt={'profile'} width='30' height='30'></Image>
              </Link>

              <Typography variant="h6" sx={{ marginLeft: 3, color: 'black', backgroundColor: '#08D354', padding: 1, borderRadius: 2, width: 100, textAlign: 'center' }}>
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
