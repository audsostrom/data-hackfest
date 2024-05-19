// components/Navbar.js
import React from 'react';
import { AppBar, Toolbar, IconButton, Typography, Drawer, List, ListItem, ListItemText, Box } from '@mui/material';
import Link from 'next/link';
import Image from 'next/image';
import Logo from '../../../../public/images/logo.svg';
import Home from '../../../../public/images/home.svg';
import Movies from '../../../../public/images/movies.gif';
import Friends from '../../../../public/images/people.svg';
import Bucket from '../../../../public/images/bucket.svg';
import SearchBar from "../search-bar";
import './landing.css'
import { getServerAuthSession } from '~/server/auth';
import { redirect } from 'next/navigation';


const Landing = async () => {
   const session = await getServerAuthSession();
   if (session) {
      redirect('/home');

   }
  return (
   <div className='landing-container'
   >
      
      <div>
     <div className="wave"></div>
     <div className="wave"></div>
     <div className="wave"></div>
     <div className='landing-content'>
      <div>
         <div className='title'>
            <Image className="rotate" src={Logo} alt={'logo'} height={100} width={100}></Image>
            <div>CineMap</div>

         </div>
         <div className='tagline'>
            Why waste time searching when you can be watching? Let CineMap do the work and enjoy your perfect movie night, every time.
         </div>
         <div className='tagline-2'>
            We leverage advanced AI to understand your unique tastes and preferences. Our app curates a selection of movies tailored just for you, saving you hours of indecision. Simply input your mood, favorite genres, or recent likes, and we'll present a short list of perfect picks.
         </div>

         <Link
              href={session ? "/api/auth/signout" : "/api/auth/signin"}
              
            >
            <div className='login-button-2'>{session ? "Sign out" : "Sign in"}</div>
         </Link>


      </div>

     </div>
      </div>
      <Image className='image-right-side' src={Movies} alt='movies' height='400' width='400'/>
   
   </div>
  );
};

export default Landing;
