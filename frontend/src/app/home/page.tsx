import './home.css';
import Image from 'next/image';
import Rows from '../../../public/images/row.jpeg';
import Profile from '../../../public/images/profile.svg';
import Question from '../../../public/images/question.svg';
import Navbar from '../_components/navbar/navbar';
import { getServerSession } from 'next-auth';
import { authOptions } from '~/server/auth';
import { redirect } from 'next/navigation';
import Link from 'next/link';


export default async function Home() {
  const session = await getServerSession(authOptions);
  console.log(session);
  if (!session) {
    redirect('/');
  }


  return (
    <>
            <div className='home-container'>
      {

        /**
               <h1 >Home Page</h1>
      <p style={{ fontFamily: 'Gurajada', fontSize: 70 }}>Welcome to the Home page!</p>
      <div style={{ fontFamily: 'DMSans' }}>Testing font imports</div>
         */
      }
      <div className='left-side'>
        <div className='profile-section'>
          <div className="banner">
          </div>
          {
            session.user.image ? <Image className='logo' src={session.user.image} alt={'profile'} width='60' height='60'></Image>
            : <Image className='logo' src={Profile} alt={'profile'} width='60' height='60'></Image>

          }
          
          <div className='all-about-you'>{session.user.name}</div>
          <div className='username' >{session.user.email}</div>
          <div className='profile-stats'>
            <div className='movies-seen'>
              <div>18</div>
              <div>Movies Seen</div>
            </div>
            <div className='friends-num'>
              <div>18</div>
              <div>Friends</div>
            </div>
            <div className='bucketlists-num'>
              <div>18</div>
              <div>Bucket Lists</div>
            </div>

          </div>
        </div>
        <div className='friend-activity'>
          <div className='all-about-you'>Friend Activity</div>
          <ol className='activity-section'>
            <li>Riley watched to Kill a MockingBird</li>
            <li>Riley watched to Kill a MockingBird</li>
            <li>Riley watched to Kill a MockingBird</li>
            <li>Riley watched to Kill a MockingBird</li>
          </ol>
        </div>
      </div>
      <div className='middle-options'>
        <div className='middle-title'>Find Your Next Movie</div>
        <div className='middle-question'>Are you watching by yourself?</div>
        <div className='question-row'>
          <Link href='solo-results'>
            <div>Pick for me!</div>
          
          </Link>
          <Image className='question' src={Question} alt={'question'} width='20' height='20'/>
        </div>
        <div className='question-row'>
            <div>Take a quiz</div>
            <Image className='question' src={Question} alt={'question'} width='20' height='20'/>

        </div>
        <div className='middle-question'>Are you watching with a group?</div>
        <div className='question-row'>
          <div>Pick for us!</div>
          <Image className='question' src={Question} alt={'question'} width='20' height='20'/>
        </div>
        <div className='question-row'>
          <div>Work together on a quiz</div>
          <Image className='question' src={Question} alt={'question'} width='20' height='20'/>
        </div>
      </div>
      <div className='right-side'>
        <div className='all-about-you'>All About You</div>
        <div className='all-about-you-subheader'>You're subscribed to</div>
        <div className='streaming'>
          <div className='streaming-button'>Netflix</div>
          <div className='streaming-button'>Netflix</div>
          <div className='streaming-button'>Netflix</div>
          <div className='streaming-button'>Netflix</div>
        </div>
        <div className='all-about-you-subheader'>You love these genres</div>
        <div>Scary, comedy, drama, romance, cartoons</div>

      </div>

    </div>
    
    </>
  );
}


