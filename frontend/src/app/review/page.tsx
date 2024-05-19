import './review.css';
import Image from 'next/image';
import { useSession } from 'next-auth/react';
import { getServerSession } from 'next-auth';
import { authOptions } from '~/server/auth';
import { redirect } from 'next/navigation';
import Navbar from '../_components/navbar/navbar';


export default async function Profile() {
  const session = await getServerSession(authOptions);
  console.log(session);
  if (!session) {
    redirect('/login');
  }

  return (
   <>
   <Navbar></Navbar>
       <div className='profile-container'>
     
    </div>
   </>

  );
}


