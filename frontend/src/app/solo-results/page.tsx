import './results.css';
import Image from 'next/image';
import { useSession } from 'next-auth/react';
import { getServerSession } from 'next-auth';
import { authOptions } from '~/server/auth';
import { redirect } from 'next/navigation';
import Navbar from '../_components/navbar/navbar';


export default async function Results() {
  const session = await getServerSession(authOptions);
  console.log('why');
  if (!session) {
    redirect('/login');
  }


  /** testing flask */
  const response = await fetch("http://127.0.0.1:5000/getmovieinfo", {
    method: 'POST',
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({year: '1990', title: 'Toy Story'})
  });
  const result = await response.json();
  console.log(result)

  return (
   <>
   <Navbar></Navbar>
    <div className='results-container'>
      Results
    </div>
   </>
  );
}


