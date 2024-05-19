
import './results.css';
import Image from 'next/image';
import { useSession } from 'next-auth/react';
import { getServerSession } from 'next-auth';
import { authOptions } from '~/server/auth';
import { redirect } from 'next/navigation';
import Loading from '../../../public/images/loading.gif';
import Navbar from '../_components/navbar/navbar';


export default async function Results() {
  const session = await getServerSession(authOptions);
  console.log('why');
  if (!session) {
    redirect('/login');
  }


  /** testing flask */

  const response = await fetch("http://127.0.0.1:5000/recommend", {
    method: 'POST',
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({user_id: 42, query: 'Action film with drama and intense fighting with a serious plot'})
  });
  const result = await response.json();

  console.log(result)
  /**
    const movieData = await fetch("http://127.0.0.1:5000/getmovieinfo", {
    method: 'POST',
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({title: 'hi'})
  });
   */


  const movieDataresult =[{'name': 'Vampires', 'streaming_platforms': 'Apple TV+, Amazon Prime Video', 'image': 'https://m.media-amazon.com/images/M/MV5BMjA4MTU2NzIzMV5BMl5BanBnXkFtZTcwNTk4MzMyMg@@._V1_FMjpg_UX1000_.jpg'}, {'name': 'Pandorum', 'streaming_platforms': 'DIRECTV, ROW8', 'image': 'https://m.media-amazon.com/images/M/MV5BMzkxNDg3NTY1MV5BMl5BanBnXkFtZTcwNjE3MTI0Mg@@._V1_FMjpg_UX1000_.jpg'}]
  // const movieDataresult = await movieData.json();
  console.log(movieDataresult)




  return (
   <>
   <Navbar></Navbar>
   {!result && 
   
    <div className='loading-container'>
      <h1 className='section-header'>Gathering Movies For You</h1>
      <div style={{marginBottom: 60, fontSize: 30}}>Please sit tight!</div>
      <Image alt={'loading'}src={Loading} height={100} width={100}></Image>
    
    </div>}
    {result && <div className='results-container'>

    {result && <div>
      <h1 className='section-header'>Here’s What We Think You’d Like</h1>
      <div>
        {result.map((movie: any, i: any) => (
          <div key={movie.movieId}>
            <h2>{movie.title}</h2>
            <p><strong>Genres:</strong> {movie.genres.replaceAll("|", ', ')}</p>
            <p><strong>Streaming Platforms:</strong> {movieDataresult[i]?.streaming_platforms}</p>
          </div>
        ))}
      </div>
    </div>
    }

    </div>}
   </>
  );
}



