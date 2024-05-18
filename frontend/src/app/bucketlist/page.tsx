import './bucketlist.css';
import Image from 'next/image';
import { useSession } from 'next-auth/react';
import { getServerSession } from 'next-auth';
import { authOptions } from '~/server/auth';
import { redirect } from 'next/navigation';
import Navbar from '../_components/navbar/navbar';


export default async function BucketList() {
  const session = await getServerSession(authOptions);
  console.log(session);
  if (!session) {
    redirect('/login');
  }

  return (
    <>
    <Navbar></Navbar>
    <div className='bucketlist-container'>
      Bucketlist
    </div>
    </>

  );
}

